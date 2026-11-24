#ifdef INPUT_UPLOAD
ByteAddressBuffer source : register(t0);
#else
RWByteAddressBuffer source : register(u0);
#endif
RWByteAddressBuffer data : register(u1);
RWByteAddressBuffer packed : register(u2);

cbuffer Config : register(b0) {
    uint count, input_offset, input_stride, action_offset;
    uint action_bytes, scratch_stride, template_bytes, patch_count;
    uint scratch_lo, scratch_hi, va_lo, va_hi;
    uint io_bytes;
};

uint2 offset_address(uint lo, uint hi, uint offset) {
    uint sum = lo + offset;
    return uint2(sum, hi + uint(sum < lo));
}

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= count)
        return;
    uint src = input_offset + id.x * input_stride;
    uint dst = id.x * scratch_stride;
    uint template_offset = count * scratch_stride;
    uint patch_offset = template_offset + template_bytes;
    for (uint i = 8; i < template_bytes; i += 4)
        data.Store(dst + i, data.Load(template_offset + i));
    data.Store2(dst, offset_address(scratch_lo, scratch_hi, dst + io_bytes));
    for (uint j = 0; j < patch_count; ++j) {
        uint2 patch = data.Load2(patch_offset + j * 8);
        data.Store(dst + patch.y, source.Load(src + patch.x));
    }

    uint record = id.x * (8 + action_bytes);
    packed.Store2(record, offset_address(va_lo, va_hi, dst));
    for (uint k = 0; k < action_bytes; k += 4)
        packed.Store(record + 8 + k, source.Load(src + action_offset + k));
}
