#include "DRMNvidiaColor.h"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

#include "backend.h"
#include "color_helpers.h"
#include "rendervulkan.hpp"

gamescope::ConVar<bool> cv_drm_debug_disable_nv_color_mgmt( "drm_debug_disable_nv_color_mgmt", false, "NVIDIA plane colour management chicken bit. (Forces the NV_* colour pipeline off, which means falling back to composition)" );
gamescope::ConVar<float> cv_drm_nv_reference_white_nits( "drm_nv_reference_white_nits", 10000.0f, "Nits that map to 1.0 in nvidia-drm's linear blending space. That space is PQ-absolute, so 1.0 is PQ peak: 10000. Only change this if SDR planes land at the wrong brightness on an HDR output." );

// struct drm_color_lut / struct drm_color_ctm from <drm_mode.h>, redeclared so
// this does not depend on which libdrm headers happen to be in the include path.
struct nv_color_lut_entry_t
{
	uint16_t red, green, blue, reserved;
};
static_assert( sizeof( nv_color_lut_entry_t ) == 8, "drm_color_lut ABI mismatch" );

// NV_PLANE_BLEND_CTM takes a struct drm_color_ctm_3x4, not the 3x3
// drm_color_ctm: nvidia-drm-crtc.c validates the blob with
// nv_drm_atomic_replace_property_blob_from_id( ..., sizeof( struct
// drm_color_ctm_3x4 ), ... ) and ctm_3x4_to_csc() indexes it as matrix[y*4 + x].
// A 3x3 blob is rejected outright, and the atomic test then fails with -EINVAL.
struct nv_color_ctm_t
{
	uint64_t matrix[12];
};
static_assert( sizeof( nv_color_ctm_t ) == 96, "drm_color_ctm_3x4 ABI mismatch" );

uint64_t nv_s31_32_from_float( float flValue )
{
	// Magnitude only; callers that need a sign set bit 63 themselves.
	double dMagnitude = std::fabs( double( flValue ) );
	// Keep well inside the 63-bit magnitude field. Nothing we feed this is
	// anywhere near the limit, so clamping rather than overflowing is fine.
	dMagnitude = std::min( dMagnitude, 65536.0 );
	return uint64_t( dMagnitude * double( k_ulNvS31_32One ) );
}

static nv_color_ctm_t nv_ctm_from_mat3( const glm::mat3 &mMatrix )
{
	nv_color_ctm_t ctm{};
	for ( int nRow = 0; nRow < 3; nRow++ )
	{
		for ( int nCol = 0; nCol < 3; nCol++ )
		{
			// glm is column-major, so this is [column][row].
			const float flValue = mMatrix[nCol][nRow];
			uint64_t ulEntry = nv_s31_32_from_float( flValue );
			if ( flValue < 0.0f )
				ulEntry |= uint64_t( 1 ) << 63;
			// Row-major 3x4; the fourth column is a constant offset per row,
			// left at zero for a pure matrix.
			ctm.matrix[ nRow * 4 + nCol ] = ulEntry;
		}
	}
	return ctm;
}

// The sRGB EOTF as a LUT for NV_PLANE_DEGAMMA_LUT. NVIDIA's degamma TF enum only
// has Default/Linear/PQ, so SDR planes need the curve supplied this way. Matches
// AMDGPU_TRANSFER_FUNCTION_SRGB_EOTF, which is what colorspace_to_plane_degamma_tf()

std::shared_ptr<gamescope::BackendBlob> nv_get_degamma_lut_blob( uint32_t uEntries )
{
	static std::shared_ptr<gamescope::BackendBlob> s_pBlob;
	static uint32_t s_uCachedEntries = 0;

	if ( !uEntries || uEntries > 16384 )
		return nullptr;

	if ( s_pBlob && s_uCachedEntries == uEntries )
		return s_pBlob;

	std::vector<nv_color_lut_entry_t> lut( uEntries );
	for ( uint32_t i = 0; i < uEntries; i++ )
	{
		const float flInput = float( i ) / float( uEntries - 1 );
		const float flLinear = srgb_to_linear( flInput );
		const uint16_t uValue = uint16_t( std::clamp( flLinear, 0.0f, 1.0f ) * 65535.0f + 0.5f );
		lut[i].red = lut[i].green = lut[i].blue = uValue;
		lut[i].reserved = 0;
	}

	const uint8_t *pBegin = reinterpret_cast<const uint8_t *>( lut.data() );
	const uint8_t *pEnd = pBegin + lut.size() * sizeof( nv_color_lut_entry_t );

	s_pBlob = GetBackend()->CreateBackendBlob( typeid( nv_color_lut_entry_t ), std::span<const uint8_t>( pBegin, pEnd ) );
	s_uCachedEntries = uEntries;
	return s_pBlob;
}


std::shared_ptr<gamescope::BackendBlob> nv_get_2020_from_709_ctm_blob()
{
	static std::shared_ptr<gamescope::BackendBlob> s_pBlob;
	if ( !s_pBlob )
	{
		const nv_color_ctm_t ctm = nv_ctm_from_mat3( k_2020_from_709 );
		s_pBlob = GetBackend()->CreateBackendBlob( ctm );
	}
	return s_pBlob;
}

nv_plane_color_config_t nv_plane_color_config_for( GamescopeAppTextureColorspace eColorspace, bool bOutputHDR )
{
	nv_plane_color_config_t config{};

	// 1.0 in nvidia-drm's blending space. Unlike AMD's FP16 pipeline, where 1.0 is
	// 80 nits so scRGB drops straight in, NVIDIA's space is PQ-absolute: the PQ
	// EOTF it builds the degamma surface from is "normalized to 1.0" against PQ
	// peak (create_drm_degamma_surface(), nvidia-drm-crtc.c), so 1.0 is 10000 nits
	// and everything else has to be scaled down into it.
	const float flBlendSpaceNits = std::max( cv_drm_nv_reference_white_nits.Get(), 1.0f );

	// scRGB defines 1.0 as 80 nits by specification.
	static constexpr float k_flScRGBWhiteNits = 80.0f;

	const float flSDRScale = bOutputHDR
		? g_ColorMgmt.pending.flSDROnHDRBrightness / flBlendSpaceNits
		: 1.0f;
	const float flScRGBScale = bOutputHDR
		? k_flScRGBWhiteNits / flBlendSpaceNits
		: 1.0f;

	switch ( eColorspace )
	{
		case GAMESCOPE_APP_TEXTURE_COLORSPACE_HDR10_PQ:
			// Already BT.2020 and already at absolute luminance; nothing to scale
			// and nothing to matrix.
			config.ulInputColorspace = NV_DRM_INPUT_COLOR_SPACE_BT2100_PQ;
			config.ulDegammaTF       = NV_DRM_TRANSFER_FUNCTION_PQ;
			break;

		case GAMESCOPE_APP_TEXTURE_COLORSPACE_SCRGB:
		case GAMESCOPE_APP_TEXTURE_COLORSPACE_PASSTHRU:
			// Already linear, so no curve: these are the two that
			// colorspace_to_plane_degamma_tf() gives an IDENTITY degamma. Scaling is
			// still needed, because scRGB's 1.0 is 80 nits while the blending space's
			// 1.0 is PQ peak.
			config.ulInputColorspace = NV_DRM_INPUT_COLOR_SPACE_SCRGB_LINEAR;
			config.ulDegammaTF       = NV_DRM_TRANSFER_FUNCTION_LINEAR;
			config.flMultiplier      = flScRGBScale;
			config.bNeeds2020Matrix  = bOutputHDR;
			break;

		case GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB:
		case GAMESCOPE_APP_TEXTURE_COLORSPACE_LINEAR:
		default:
			// LINEAR is sRGB-encoded content, despite the name. The composite path
			// gets to treat it as linear because it samples through an sRGB image
			// view; scanout has no image view, so the curve has to be applied here
			// explicitly. colorspace_to_plane_degamma_tf() maps both of these to
			// SRGB_EOTF for exactly that reason, and its comment says so.
			config.bNeedsDegammaLut = true;
			config.flMultiplier     = flSDRScale;
			config.bNeeds2020Matrix = bOutputHDR;
			break;
	}

	return config;
}
