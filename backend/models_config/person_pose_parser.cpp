#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cmath>

extern "C" bool NvDsInferParseYoloV8Pose(
    const std::vector<NvDsInferLayerInfo>& layers,
    const NvDsInferNetworkInfo& network,
    const NvDsInferParseDetectionParams& parameters,
    std::vector<NvDsInferObjectDetectionInfo>& objects)
{
    if (layers.empty() || parameters.perClassPreclusterThreshold.empty()) return false;
    const auto& layer = layers.front();
    const auto& dimensions = layer.inferDims;
    const unsigned offset = dimensions.numDims == 3 ? 1 : 0;
    if (dimensions.numDims < 2 || dimensions.d[offset] != 56) return false;
    const unsigned anchors = dimensions.d[offset + 1];
    const auto* values = static_cast<const float*>(layer.buffer);
    if (!values || layer.dataType != FLOAT) return false;
    for (unsigned anchor = 0; anchor < anchors; ++anchor) {
        const float confidence = values[4 * anchors + anchor];
        if (!std::isfinite(confidence) || confidence < parameters.perClassPreclusterThreshold[0]) continue;
        const float centerX = values[anchor];
        const float centerY = values[anchors + anchor];
        const float width = values[2 * anchors + anchor];
        const float height = values[3 * anchors + anchor];
        if (!std::isfinite(centerX) || !std::isfinite(centerY) || !std::isfinite(width) || !std::isfinite(height)) continue;
        const float left = std::max(0.0f, centerX - width / 2);
        const float top = std::max(0.0f, centerY - height / 2);
        const float right = std::min(float(network.width), centerX + width / 2);
        const float bottom = std::min(float(network.height), centerY + height / 2);
        if (right - left < 1 || bottom - top < 1) continue;
        NvDsInferObjectDetectionInfo object{};
        object.classId = 0;
        object.detectionConfidence = confidence;
        object.left = left;
        object.top = top;
        object.width = right - left;
        object.height = bottom - top;
        objects.push_back(object);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloV8Pose);
