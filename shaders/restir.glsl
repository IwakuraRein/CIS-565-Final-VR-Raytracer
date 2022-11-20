#include "host_device.h"

const int ReservoirSize = 32;

float resvToScalar(vec3 x) {
    return length(x);
}

void resvReset(inout Reservoir resv) {
    resv.num = 0;
    resv.weight = 0;
}

bool resvInvalid(Reservoir resv) {
    return isnan(resv.weight) || resv.weight < 0.0;
}

void resvCheckValidity(inout Reservoir resv) {
    if (resvInvalid(resv)) {
        resvReset(resv);
    }
}

void resvUpdate(inout Reservoir resv, LightSample newSample, float newWeight, float r) {
    resv.weight += newWeight;
    resv.num += 1;
    if (r * resv.weight < newWeight) {
        resv.lightSample = newSample;
    }
}

void resvMerge(inout Reservoir resv, Reservoir rhs, float r) {
    resv.weight += rhs.weight;
    resv.num += rhs.num;
    if (r * resv.weight < rhs.weight) {
        resv.lightSample = rhs.lightSample;
    }
}

void resvPreClampedMerge(inout Reservoir resv, Reservoir rhs, float r, int clamp) {
    if (rhs.num > 0 && resv.num > 0 && rhs.num > (clamp - 1) * resv.num) {
        rhs.weight *= float(clamp - 1) * float(resv.num) / float(rhs.num);
        rhs.num = (clamp - 1) * resv.num;
    }
    resvMerge(resv, rhs, r);
}

void resvPreClampedMerge20(inout Reservoir resv, Reservoir rhs, float r) {
    if (rhs.num > 0 && resv.num > 0 && rhs.num > 19 * resv.num) {
        rhs.weight *= float(19) * float(resv.num) / float(rhs.num);
        rhs.num = 19 * resv.num;
    }
    resvMerge(resv, rhs, r);
}

// 32bit Li, 32bit direction, 24bit weight, 16bit num, 24bit dist
// untested
uvec4 encodeReservoir(Reservoir resv) {
    uvec4 pack;
    resv.num = resv.num & 0xFFFF;
    pack.x = packUnormYCbCr(resv.lightSample.Li);
    pack.y = compress_unit_vec(resv.lightSample.wi);
    pack.z = resv.num >> 8;
    pack.z += floatBitsToUint(resv.weight) & 0xFFFFFF00;
    pack.w = resv.num << 24;
    pack.w += floatBitsToUint(resv.lightSample.dist) >> 8;
    return pack;
}

Reservoir decodeReservoir(uvec4 pack) {
    Reservoir resv;
    resv.lightSample.Li = unpackUnormYCbCr(pack.x);
    resv.lightSample.wi = decompress_unit_vec(pack.y);
    resv.weight = uintBitsToFloat(pack.z & 0xFFFFFF00);
    resv.lightSample.dist = uintBitsToFloat(pack.w << 8);
    resv.num = (pack.z & 0xFF) + (pack.w >> 24);
    return resv;
}