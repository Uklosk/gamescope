#pragma once

///////////////////////////////////////////////////////////////////////////////
// nvidia-drm vendor colour pipeline.
//
// nvidia-drm exposes its own per-plane colour management properties rather than
// the AMD_PLANE_* ones the rest of this file was written against, and the two
// are not equivalent. NVIDIA has no 3D LUT at all, and the tone-mapping LUT and
// LMS matrices exist only on the primary plane, so gamescope's shaper + 3D LUT
// colour management cannot be ported across as-is.
//
// What every NVIDIA plane does expose is enough to linearise its contents,
// scale them, matrix them into the output primaries and blend there:
//
//   NV_INPUT_COLORSPACE -> NV_PLANE_DEGAMMA_TF / NV_PLANE_DEGAMMA_LUT
//     -> NV_PLANE_DEGAMMA_MULTIPLIER -> NV_PLANE_BLEND_CTM -> blend
//
// and per CRTC, NV_CRTC_REGAMMA_TF re-encodes the blended result at scanout.
// That is exactly what is needed to put an SDR overlay on top of an HDR10 plane
// without a full composite, which is the case this exists for.
//
// Property and enum definitions mirror the NVIDIA open kernel modules:
//   enum nv_drm_transfer_function  - kernel-open/nvidia-drm/nv_drm_common_ioctl.h
//   enum nv_drm_input_color_space  - kernel-open/nvidia-drm/nvidia-drm-priv.h
//   NV_DRM_S31_32_ONE              - kernel-open/nvidia-drm/nvidia-drm-helper.h
///////////////////////////////////////////////////////////////////////////////

#include <cstdint>
#include <memory>

#include "convar.h"
#include "gamescope_shared.h"

namespace gamescope { class BackendBlob; }

extern gamescope::ConVar<bool> cv_drm_debug_disable_nv_color_mgmt;
extern gamescope::ConVar<float> cv_drm_nv_reference_white_nits;

enum nv_drm_transfer_function : uint64_t
{
	NV_DRM_TRANSFER_FUNCTION_DEFAULT = 0,
	NV_DRM_TRANSFER_FUNCTION_LINEAR  = 1,
	NV_DRM_TRANSFER_FUNCTION_PQ      = 2,
	// The enum in nv_drm_common_ioctl.h continues past this, but nvidia-drm
	// only implements the three above; the rest are tegradisp-drm only.
};

enum nv_drm_input_color_space : uint64_t
{
	NV_DRM_INPUT_COLOR_SPACE_NONE         = 0,
	NV_DRM_INPUT_COLOR_SPACE_SCRGB_LINEAR = 1,
	NV_DRM_INPUT_COLOR_SPACE_BT2100_PQ    = 2,
};

// NV_PLANE_DEGAMMA_MULTIPLIER and NV_CRTC_REGAMMA_DIVISOR are S31.32
// sign-magnitude. The kernel rejects the sign bit on both of them.
static constexpr uint64_t k_ulNvS31_32One = uint64_t( 1 ) << 32;

struct nv_plane_color_config_t
{
	uint64_t ulInputColorspace = NV_DRM_INPUT_COLOR_SPACE_NONE;
	uint64_t ulDegammaTF       = NV_DRM_TRANSFER_FUNCTION_DEFAULT;
	// NVIDIA has no gamma/sRGB degamma TF, so SDR content needs the curve as a LUT.
	bool     bNeedsDegammaLut  = false;
	// Scales the plane into the blend space, where 1.0 is
	// cv_drm_nv_reference_white_nits.
	float    flMultiplier      = 1.0f;
	bool     bNeeds2020Matrix  = false;
};

// Converts to the S31.32 sign-magnitude encoding NV_PLANE_DEGAMMA_MULTIPLIER and
// NV_CRTC_REGAMMA_DIVISOR use. Magnitude only; the sign bit is never set here
// because the kernel rejects negative values on both properties.
uint64_t nv_s31_32_from_float( float flValue );

// How a plane carrying this colorspace should be programmed for an output that
// is or is not HDR.
nv_plane_color_config_t nv_plane_color_config_for( GamescopeAppTextureColorspace eColorspace, bool bOutputHDR );

// Cached blobs. Both are constant for the life of the process, so these hand back
// the same blob every call.
std::shared_ptr<gamescope::BackendBlob> nv_get_degamma_lut_blob( uint32_t uEntries );
std::shared_ptr<gamescope::BackendBlob> nv_get_2020_from_709_ctm_blob();
