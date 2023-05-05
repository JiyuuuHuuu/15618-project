#ifndef SEQ_H
#define SEQ_H

#include "constants.h"

void seqLauncher(uchar4 *d_out, int w, int h, particle *particles, tail *tails, int *idx_holder, float t, const particle *schedule);

#endif