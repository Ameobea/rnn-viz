export const RESOLUTION = 80;
export const DOMAIN = [-1, 1];
export const RANGE = [-1, 1];
export const VOXEL_SIZE = (DOMAIN[1] - DOMAIN[0]) / RESOLUTION;

// TODO: Tune
export const MAX_VOXEL_COUNT = RESOLUTION * RESOLUTION * RESOLUTION;
