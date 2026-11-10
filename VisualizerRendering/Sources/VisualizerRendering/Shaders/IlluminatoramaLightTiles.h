// IlluminatoramaLightTiles.h — the tiled light masks' layout (Daydream DH-1625), shared by the kernel that
// builds them (`illumi_light_tiles`, IlluminatoramaLightBounds.metal) and the one that reads them
// (`illumi_lighting`). One mask per `kLightTileSize`² pixel tile of the INTERNAL frame, `kLightTileWords`
// 32-bit words: bit n = light n may reach a surface in this tile, lights numbered points, then spots, then
// area lights. A light past `kLightTileBits` has no bit and is always evaluated.
#ifndef ILLUMINATORAMA_LIGHT_TILES_H
#define ILLUMINATORAMA_LIGHT_TILES_H

#define kLightTileSize  16u
#define kLightTileWords 4u
#define kLightTileBits  (kLightTileWords * 32u)

/// The tile's mask words, by value (four register reads for the whole pixel).
struct LightTileMask { uint w[kLightTileWords]; };

/// Whether light `n` may reach this tile. `n >= kLightTileBits` ⇒ true (no bit to consult).
static inline bool lightTileHas(LightTileMask m, uint n) {
    return n >= kLightTileBits || ((m.w[n >> 5u] >> (n & 31u)) & 1u) != 0u;
}

#endif
