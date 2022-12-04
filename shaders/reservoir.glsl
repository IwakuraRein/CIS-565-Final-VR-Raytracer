#ifndef RESERVOIR_GLSL
#define RESERVOIR_GLSL

#include "host_device.h"

float resvToScalar(vec3 x) {
    //return length(x);
    return luminance(x);
}

void resvReset(inout DirectReservoir resv) {
    resv.num = 0;
    resv.weight = 0;
}

void resvReset(inout IndirectReservoir resv) {
    resv.num = 0;
    resv.weight = 0;
    resv.bigW = 0;
}

void resvUpdateBigW(inout IndirectReservoir resv, float pHat) {
    resv.bigW = resv.weight / (float(resv.num) * pHat);
}

bool resvInvalid(DirectReservoir resv) {
    return isnan(resv.weight) || resv.weight < 0.0;
}

bool resvInvalid(IndirectReservoir resv) {
    return isnan(resv.weight) || resv.weight < 0.0;
}

void resvCheckValidity(inout DirectReservoir resv) {
    if (resvInvalid(resv)) {
        resvReset(resv);
    }
}

void resvCheckValidity(inout IndirectReservoir resv) {
    if (resvInvalid(resv)) {
        resvReset(resv);
    }
}

void resvUpdate(inout DirectReservoir resv, LightSample newSample, float newWeight, float r) {
    resv.weight += newWeight;
    resv.num += 1;
    if (r * resv.weight < newWeight) {
        resv.lightSample = newSample;
    }
}

void resvUpdate(inout IndirectReservoir resv, GISample newSample, float newWeight, float r) {
    resv.weight += newWeight;
    resv.num += 1;
    if (r * resv.weight < newWeight) {
        resv.giSample = newSample;
    }
}

void resvMerge(inout IndirectReservoir resv, IndirectReservoir rhs, float pHat, float r) {
    uint num = resv.num;
    resvUpdate(resv, rhs.giSample, pHat * rhs.bigW * float(rhs.num), r);
    resv.num = num + rhs.num;
}

void resvMerge(inout DirectReservoir resv, DirectReservoir rhs, float r) {
    resv.weight += rhs.weight;
    resv.num += rhs.num;
    if (r * resv.weight < rhs.weight) {
        resv.lightSample = rhs.lightSample;
    }
}

void resvMerge(inout IndirectReservoir resv, IndirectReservoir rhs, float r) {
    resv.weight += rhs.weight;
    resv.num += rhs.num;
    if (r * resv.weight < rhs.weight) {
        resv.giSample = rhs.giSample;
    }
}

void resvPreClampedMerge(inout DirectReservoir resv, DirectReservoir rhs, float r, int clamp) {
    if (rhs.num > 0 && resv.num > 0 && rhs.num > (clamp - 1) * resv.num) {
        rhs.weight *= float(clamp - 1) * float(resv.num) / float(rhs.num);
        rhs.num = (clamp - 1) * resv.num;
    }
    resvMerge(resv, rhs, r);
}

void resvPreClampedMerge(inout IndirectReservoir resv, IndirectReservoir rhs, float r, int clamp) {
    if (rhs.num > 0 && resv.num > 0 && rhs.num > (clamp - 1) * resv.num) {
        rhs.weight *= float(clamp - 1) * float(resv.num) / float(rhs.num);
        rhs.num = (clamp - 1) * resv.num;
    }
    resvMerge(resv, rhs, r);
}

void resvPreClampedMerge20(inout DirectReservoir resv, DirectReservoir rhs, float r) {
    if (rhs.num > 0 && resv.num > 0 && rhs.num > 19 * resv.num) {
        rhs.weight *= float(19) * float(resv.num) / float(rhs.num);
        rhs.num = 19 * resv.num;
    }
    resvMerge(resv, rhs, r);
}

void resvPreClampedMerge20(inout IndirectReservoir resv, IndirectReservoir rhs, float r) {
    if (rhs.num > 0 && resv.num > 0 && rhs.num > 19 * resv.num) {
        rhs.weight *= float(19) * float(resv.num) / float(rhs.num);
        rhs.num = 19 * resv.num;
    }
    resvMerge(resv, rhs, r);
}

void resvClamp(inout DirectReservoir resv, int clamp) {
    if (resv.num > clamp) {
        resv.weight *= float(clamp) / float(resv.num);
        resv.num = clamp;
    }
}

void resvClamp(inout IndirectReservoir resv, int clamp) {
    if (resv.num > clamp) {
        resv.weight *= float(clamp) / float(resv.num);
        resv.num = clamp;
    }
}

// 32bit Li, 32bit direction, 24bit weight, 16bit num, 24bit dist
// untested
// uvec4 encodeReservoir(DirectReservoir resv) {
//     uvec4 pack;
//     resv.num = resv.num & 0xFFFF;
//     pack.x = packUnormYCbCr(resv.lightSample.Li);
//     pack.y = compress_unit_vec(resv.lightSample.wi);
//     pack.z = resv.num >> 8;
//     pack.z += floatBitsToUint(resv.weight) & 0xFFFFFF00;
//     pack.w = resv.num << 24;
//     pack.w += floatBitsToUint(resv.lightSample.dist) >> 8;
//     return pack;
// }

// DirectReservoir decodeReservoir(uvec4 pack) {
//     DirectReservoir resv;
//     resv.lightSample.Li = unpackUnormYCbCr(pack.x);
//     resv.lightSample.wi = decompress_unit_vec(pack.y);
//     resv.weight = uintBitsToFloat(pack.z & 0xFFFFFF00);
//     resv.lightSample.dist = uintBitsToFloat(pack.w << 8);
//     resv.num = (pack.z & 0xFF) + (pack.w >> 24);
//     return resv;
// }

#endif
