#include "host_device.h"

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