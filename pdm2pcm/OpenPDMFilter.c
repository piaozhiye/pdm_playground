/**
 *******************************************************************************
 * @file    OpenPDMFilter.c
 * @author  CL
 * @version V1.0.0
 * @date    9-September-2015
 * @brief   Open PDM audio software decoding Library.   
 *          This Library is used to decode and reconstruct the audio signal
 *          produced by ST MEMS microphone (MP45Dxxx, MP34Dxxx). 
 *******************************************************************************
 * @attention
 *
 * <h2><center>&copy; COPYRIGHT 2018 STMicroelectronics</center></h2>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************
 */


/* Includes ------------------------------------------------------------------*/

#include "OpenPDMFilter.h"


/* Variables -----------------------------------------------------------------*/

int64_t div_const = 0;
int64_t sub_const = 0;
int64_t sinc[DECIMATION_MAX * SINCN];
int64_t coef[SINCN][DECIMATION_MAX];
#ifdef USE_LUT
int32_t lut[256][DECIMATION_MAX / 8][SINCN];
#endif


/* Functions -----------------------------------------------------------------*/

#ifdef USE_LUT
int32_t filter_table_mono_64(uint8_t *data, uint8_t sincn)
{
  return (int32_t)
    lut[data[0]][0][sincn] +
    lut[data[1]][1][sincn] +
    lut[data[2]][2][sincn] +
    lut[data[3]][3][sincn] +
    lut[data[4]][4][sincn] +
    lut[data[5]][5][sincn] +
    lut[data[6]][6][sincn] +
    lut[data[7]][7][sincn];
}
int32_t filter_table_stereo_64(uint8_t *data, uint8_t sincn)
{
  return (int32_t)
    lut[data[0]][0][sincn] +
    lut[data[2]][1][sincn] +
    lut[data[4]][2][sincn] +
    lut[data[6]][3][sincn] +
    lut[data[8]][4][sincn] +
    lut[data[10]][5][sincn] +
    lut[data[12]][6][sincn] +
    lut[data[14]][7][sincn];
}
int32_t filter_table_mono_128(uint8_t *data, uint8_t sincn)
{
  return (int32_t)
    lut[data[0]][0][sincn] +
    lut[data[1]][1][sincn] +
    lut[data[2]][2][sincn] +
    lut[data[3]][3][sincn] +
    lut[data[4]][4][sincn] +
    lut[data[5]][5][sincn] +
    lut[data[6]][6][sincn] +
    lut[data[7]][7][sincn] +
    lut[data[8]][8][sincn] +
    lut[data[9]][9][sincn] +
    lut[data[10]][10][sincn] +
    lut[data[11]][11][sincn] +
    lut[data[12]][12][sincn] +
    lut[data[13]][13][sincn] +
    lut[data[14]][14][sincn] +
    lut[data[15]][15][sincn];
}
int32_t filter_table_stereo_128(uint8_t *data, uint8_t sincn)
{
  return (int32_t)
    lut[data[0]][0][sincn] +
    lut[data[2]][1][sincn] +
    lut[data[4]][2][sincn] +
    lut[data[6]][3][sincn] +
    lut[data[8]][4][sincn] +
    lut[data[10]][5][sincn] +
    lut[data[12]][6][sincn] +
    lut[data[14]][7][sincn] +
    lut[data[16]][8][sincn] +
    lut[data[18]][9][sincn] +
    lut[data[20]][10][sincn] +
    lut[data[22]][11][sincn] +
    lut[data[24]][12][sincn] +
    lut[data[26]][13][sincn] +
    lut[data[28]][14][sincn] +
    lut[data[30]][15][sincn];
}
int32_t filter_table_R_64(uint8_t *data, uint8_t sincn)
{
  return (int32_t)
    lut[data[1]][0][sincn] +
    lut[data[3]][1][sincn] +
    lut[data[5]][2][sincn] +
    lut[data[7]][3][sincn] +
    lut[data[9]][4][sincn] +
    lut[data[11]][5][sincn] +
    lut[data[13]][6][sincn] +
    lut[data[15]][7][sincn];
}
int32_t filter_table_R_128(uint8_t *data, uint8_t sincn)
{
  return (int32_t)
    lut[data[1]][0][sincn] +
    lut[data[3]][1][sincn] +
    lut[data[5]][2][sincn] +
    lut[data[7]][3][sincn] +
    lut[data[9]][4][sincn] +
    lut[data[11]][5][sincn] +
    lut[data[13]][6][sincn] +
    lut[data[15]][7][sincn] +
    lut[data[17]][8][sincn] +
    lut[data[19]][9][sincn] +
    lut[data[21]][10][sincn] +
    lut[data[23]][11][sincn] +
    lut[data[25]][12][sincn] +
    lut[data[27]][13][sincn] +
    lut[data[29]][14][sincn] +
    lut[data[31]][15][sincn];
}
int32_t (* filter_tables_64[2]) (uint8_t *data, uint8_t sincn) = {filter_table_mono_64, filter_table_stereo_64};
int32_t (* filter_tables_128[2]) (uint8_t *data, uint8_t sincn) = {filter_table_mono_128, filter_table_stereo_128};
#endif

int64_t filter_table(uint8_t *data, uint8_t sincn, TPDMFilter_InitStruct *param)
{
  uint8_t c, i;
  uint16_t data_index = 0;
  int64_t *coef_p = &coef[sincn][0];
  int64_t F = 0;
  uint8_t decimation = param->Decimation;
  uint8_t channels = param->In_MicChannels;

  for (i = 0; i < decimation; i += 8) {
    c = data[data_index];
    F += ((c >> 7)       ) * coef_p[i    ] +
         ((c >> 6) & 0x01) * coef_p[i + 1] +
         ((c >> 5) & 0x01) * coef_p[i + 2] +
         ((c >> 4) & 0x01) * coef_p[i + 3] +
         ((c >> 3) & 0x01) * coef_p[i + 4] +
         ((c >> 2) & 0x01) * coef_p[i + 5] +
         ((c >> 1) & 0x01) * coef_p[i + 6] +
         ((c     ) & 0x01) * coef_p[i + 7];
    data_index += channels;
  }
  return F;
}

/* R-channel variant of filter_table: odd bytes (byte-interleaved stereo). */
static int64_t filter_table_R(uint8_t *data, uint8_t sincn, TPDMFilter_InitStruct *param)
{
  uint8_t c, i;
  uint16_t data_index = param->In_MicChannels - 1;
  int64_t *coef_p = &coef[sincn][0];
  int64_t F = 0;
  uint8_t decimation = param->Decimation;
  uint8_t channels = param->In_MicChannels;

  for (i = 0; i < decimation; i += 8) {
    c = data[data_index];
    F += ((c >> 7)       ) * coef_p[i    ] +
         ((c >> 6) & 0x01) * coef_p[i + 1] +
         ((c >> 5) & 0x01) * coef_p[i + 2] +
         ((c >> 4) & 0x01) * coef_p[i + 3] +
         ((c >> 3) & 0x01) * coef_p[i + 4] +
         ((c >> 2) & 0x01) * coef_p[i + 5] +
         ((c >> 1) & 0x01) * coef_p[i + 6] +
         ((c     ) & 0x01) * coef_p[i + 7];
    data_index += channels;
  }
  return F;
}

void convolve(uint32_t Signal[/* SignalLen */], unsigned short SignalLen,
              uint32_t Kernel[/* KernelLen */], unsigned short KernelLen,
              uint32_t Result[/* SignalLen + KernelLen - 1 */])
{
  uint16_t n;

  for (n = 0; n < SignalLen + KernelLen - 1; n++)
  {
    unsigned short kmin, kmax, k;
    
    Result[n] = 0;
    
    kmin = (n >= KernelLen - 1) ? n - (KernelLen - 1) : 0;
    kmax = (n < SignalLen - 1) ? n : SignalLen - 1;
    
    for (k = kmin; k <= kmax; k++) {
      Result[n] += Signal[k] * Kernel[n - k];
    }
  }
}

void Open_PDM_Filter_Init(TPDMFilter_InitStruct *Param)
{
  uint16_t i, j;
  uint64_t sum = 0;

  uint8_t decimation = Param->Decimation;

  for (i = 0; i < SINCN; i++) {
    Param->Coef[i] = 0;
    Param->CoefR[i] = 0;
    Param->bit[i] = 0;
  }
  Param->OldOut = Param->OldIn = Param->OldZ = 0;
  Param->OldOutR = Param->OldInR = Param->OldZR = 0;
  Param->LP_ALFA = (Param->LP_HZ != 0 ? (uint16_t) (Param->LP_HZ * 256 / (Param->LP_HZ + Param->Fs / (2 * 3.14159))) : 0);
  Param->HP_ALFA = (Param->HP_HZ != 0 ? (uint16_t) (Param->Fs * 256 / (2 * 3.14159 * Param->HP_HZ + Param->Fs)) : 0);

  Param->FilterLen = decimation * Param->CicOrder;
  /* CIC kernel = boxcar^CicOrder, centered in sinc[] like stock:
     conv length is CicOrder*(d-1)+1, padded (CicOrder-1)/2 zeros per end
     (CicOrder is odd). One int64 build for both orders; integer-exact. */
  {
    int64_t cur[DECIMATION_MAX * SINCN] = {0};
    int64_t nxt[DECIMATION_MAX * SINCN];
    int cur_len = decimation, k, n, m;
    int tap_shift = 0;
    uint64_t ksum = 0;
    for (n = 0; n < decimation; n++) cur[n] = 1;
    for (k = 1; k < Param->CicOrder; k++) {
      for (n = 0; n < cur_len + decimation - 1; n++) nxt[n] = 0;
      for (n = 0; n < cur_len; n++)
        for (m = 0; m < decimation; m++)
          nxt[n + m] += cur[n];
      cur_len += decimation - 1;
      for (n = 0; n < cur_len; n++) cur[n] = nxt[n];
    }
    for (n = 0; n < cur_len; n++) ksum += (uint64_t)cur[n];
    /* int64 headroom (Hogenauer W2): Z scales with the KERNEL SUM and the
       HP stage forms ~3x Z and multiplies it by 255.  Keep the sum <= 2^54
       so the fixed-point chain stays inside int64 for every supported
       order/rate (CIC9 d=128 sums to 2^63 -> shift 9); the chain is
       self-scaling (div_const follows the sum). */
    while (tap_shift < 30 && (ksum >> tap_shift) > (1ULL << 54))
      tap_shift++;
    if (tap_shift)
      for (n = 0; n < cur_len; n++)
        cur[n] = (cur[n] + (1 << (tap_shift - 1))) >> tap_shift;
    for (n = 0; n < SINCN * decimation; n++) sinc[n] = 0;
    for (n = 0; n < cur_len; n++)
      sinc[n + (Param->CicOrder - 1) / 2] = cur[n];
  }
  for(j = 0; j < Param->CicOrder; j++) {
    for (i = 0; i < decimation; i++) {
      coef[j][i] = sinc[j * decimation + i];
      sum += sinc[j * decimation + i];
    }
  }

  sub_const = sum >> 1;
  /* The tap rescaling above caps sum at 2^54, so sub_const*16 <= 2^57 fits
     int64 and plain 64-bit arithmetic would do; __int128 stays as a guard
     in case the rescaling is ever relaxed (unscaled N=9 d=128: sub_const*16
     = 2^66, and div_const itself reaches 2^47, past uint32). */
  div_const = (int64_t)((__int128)sub_const * Param->MaxVolume / 32768 / FILTER_GAIN);
  div_const = (div_const == 0 ? 1 : div_const);

#ifdef USE_LUT
  /* Look-Up Table. Only the 3-stage kernel fits the int32 LUT:
     N=9 taps need ~50 bits and use the filter_table path instead. */
  uint16_t c, d, s;
  for (s = 0; s < 3 && Param->CicOrder == 3; s++)
  {
    int64_t *coef_p = &coef[s][0];
    for (c = 0; c < 256; c++)
      for (d = 0; d < decimation / 8; d++)
        lut[c][d][s] = ((c >> 7)       ) * coef_p[d * 8    ] +
                       ((c >> 6) & 0x01) * coef_p[d * 8 + 1] +
                       ((c >> 5) & 0x01) * coef_p[d * 8 + 2] +
                       ((c >> 4) & 0x01) * coef_p[d * 8 + 3] +
                       ((c >> 3) & 0x01) * coef_p[d * 8 + 4] +
                       ((c >> 2) & 0x01) * coef_p[d * 8 + 5] +
                       ((c >> 1) & 0x01) * coef_p[d * 8 + 6] +
                       ((c     ) & 0x01) * coef_p[d * 8 + 7];
  }
#endif
}

static int16_t open_pdm_quantize_16(int64_t value)
{
  value = RoundDiv(value, div_const);
  value = SaturaLH(value, -32700, 32700);
  return (int16_t)value;
}

static int32_t open_pdm_quantize_24(int64_t value)
{
  /* N=9 chains scale value up to ~2^56 (Hogenauer W2 + volume), so
     value*256 would overflow int64.  When div_const is large enough the
     division is done first -- exact, because div_const is a power of two
     >= 512 for every supported order/rate combination. */
  if (div_const >= 512)
    value = RoundDiv(value, div_const / 256);
  else
    value = RoundDiv(value * 256, div_const);
  value = SaturaLH(value, -8388608LL, 8388607LL);
  return (int32_t)value;
}

static void open_pdm_store_sample(void *dataOut, uint32_t index,
                                  uint8_t output_bits, int64_t value)
{
  if (output_bits == 24)
    ((int32_t *)dataOut)[index] = open_pdm_quantize_24(value);
  else
    ((int16_t *)dataOut)[index] = open_pdm_quantize_16(value);
}

static void open_pdm_filter_64_common(uint8_t *data, void *dataOut,
                                       uint16_t volume,
                                       TPDMFilter_InitStruct *Param,
                                       uint8_t output_bits)
{
  uint32_t i, data_out_index;
  uint8_t channels = Param->In_MicChannels;
  uint8_t data_inc = ((DECIMATION_MAX >> 4) * channels);
  int64_t Z;
  int64_t z[SINCN];
  int k;
  int64_t ZR;
  int64_t zr[SINCN];
  int64_t OldOut, OldIn, OldZ;
  int64_t OldOutR, OldInR, OldZR;

  OldOut = Param->OldOut;
  OldIn = Param->OldIn;
  OldZ = Param->OldZ;
  OldOutR = Param->OldOutR;
  OldInR = Param->OldInR;
  OldZR = Param->OldZR;

#ifdef USE_LUT
  uint8_t j = channels - 1;
#endif

  for (i = 0, data_out_index = 0; i < Param->nSamples; i++, data_out_index += channels) {
    /* Phase k contributes at delay (CicOrder-1-k) output samples:
       Coef[k] must chain the PREVIOUS sample's Coef[k-1] (the stock
       N=3 code is the k<=2 case of this loop). */
#ifdef USE_LUT
    if (Param->CicOrder == 3) {
      for (k = 0; k < Param->CicOrder; k++)
        z[k] = filter_tables_64[j](data, k);
    } else
#endif
    {
      for (k = 0; k < Param->CicOrder; k++)
        z[k] = filter_table(data, k, Param);
    }

    Z = Param->Coef[Param->CicOrder - 2] + z[Param->CicOrder - 1] - sub_const;
    for (k = Param->CicOrder - 1; k >= 1; k--)
      Param->Coef[k] = Param->Coef[k - 1] + z[k];
    Param->Coef[0] = z[0];

    OldOut = (Param->HP_ALFA * (OldOut + Z - OldIn)) >> 8;
    OldIn = Z;
    OldZ = ((256 - Param->LP_ALFA) * OldZ + Param->LP_ALFA * OldOut) >> 8;

    Z = OldZ * volume;
    open_pdm_store_sample(dataOut, data_out_index, output_bits, Z);
    if (channels == 2) {
#ifdef USE_LUT
      if (Param->CicOrder == 3) {
        for (k = 0; k < Param->CicOrder; k++)
          zr[k] = filter_table_R_64(data, k);
      } else
#endif
      {
        for (k = 0; k < Param->CicOrder; k++)
          zr[k] = filter_table_R(data, k, Param);
      }

      ZR = Param->CoefR[Param->CicOrder - 2] + zr[Param->CicOrder - 1] - sub_const;
      for (k = Param->CicOrder - 1; k >= 1; k--)
        Param->CoefR[k] = Param->CoefR[k - 1] + zr[k];
      Param->CoefR[0] = zr[0];

      OldOutR = (Param->HP_ALFA * (OldOutR + ZR - OldInR)) >> 8;
      OldInR = ZR;
      OldZR = ((256 - Param->LP_ALFA) * OldZR + Param->LP_ALFA * OldOutR) >> 8;

      ZR = OldZR * volume;
      open_pdm_store_sample(dataOut, data_out_index + 1, output_bits, ZR);
    }
    data += data_inc;
  }

  Param->OldOut = OldOut;
  Param->OldIn = OldIn;
  Param->OldZ = OldZ;
  if (channels == 2) {
    Param->OldOutR = OldOutR;
    Param->OldInR = OldInR;
    Param->OldZR = OldZR;
  }
}

static void open_pdm_filter_128_common(uint8_t *data, void *dataOut,
                                       uint16_t volume,
                                       TPDMFilter_InitStruct *Param,
                                       uint8_t output_bits)
{
  uint32_t i, data_out_index;
  uint8_t channels = Param->In_MicChannels;
  uint8_t data_inc = ((DECIMATION_MAX >> 3) * channels);
  int64_t Z;
  int64_t z[SINCN];
  int k;
  int64_t ZR;
  int64_t zr[SINCN];
  int64_t OldOut, OldIn, OldZ;
  int64_t OldOutR, OldInR, OldZR;

  OldOut = Param->OldOut;
  OldIn = Param->OldIn;
  OldZ = Param->OldZ;
  OldOutR = Param->OldOutR;
  OldInR = Param->OldInR;
  OldZR = Param->OldZR;

#ifdef USE_LUT
  uint8_t j = channels - 1;
#endif

  for (i = 0, data_out_index = 0; i < Param->nSamples; i++, data_out_index += channels) {
    /* See open_pdm_filter_64_common for the cascade derivation. */
#ifdef USE_LUT
    if (Param->CicOrder == 3) {
      for (k = 0; k < Param->CicOrder; k++)
        z[k] = filter_tables_128[j](data, k);
    } else
#endif
    {
      for (k = 0; k < Param->CicOrder; k++)
        z[k] = filter_table(data, k, Param);
    }

    Z = Param->Coef[Param->CicOrder - 2] + z[Param->CicOrder - 1] - sub_const;
    for (k = Param->CicOrder - 1; k >= 1; k--)
      Param->Coef[k] = Param->Coef[k - 1] + z[k];
    Param->Coef[0] = z[0];

    OldOut = (Param->HP_ALFA * (OldOut + Z - OldIn)) >> 8;
    OldIn = Z;
    OldZ = ((256 - Param->LP_ALFA) * OldZ + Param->LP_ALFA * OldOut) >> 8;

    Z = OldZ * volume;
    open_pdm_store_sample(dataOut, data_out_index, output_bits, Z);
    if (channels == 2) {
#ifdef USE_LUT
      if (Param->CicOrder == 3) {
        for (k = 0; k < Param->CicOrder; k++)
          zr[k] = filter_table_R_128(data, k);
      } else
#endif
      {
        for (k = 0; k < Param->CicOrder; k++)
          zr[k] = filter_table_R(data, k, Param);
      }

      ZR = Param->CoefR[Param->CicOrder - 2] + zr[Param->CicOrder - 1] - sub_const;
      for (k = Param->CicOrder - 1; k >= 1; k--)
        Param->CoefR[k] = Param->CoefR[k - 1] + zr[k];
      Param->CoefR[0] = zr[0];

      OldOutR = (Param->HP_ALFA * (OldOutR + ZR - OldInR)) >> 8;
      OldInR = ZR;
      OldZR = ((256 - Param->LP_ALFA) * OldZR + Param->LP_ALFA * OldOutR) >> 8;

      ZR = OldZR * volume;
      open_pdm_store_sample(dataOut, data_out_index + 1, output_bits, ZR);
    }
    data += data_inc;
  }

  Param->OldOut = OldOut;
  Param->OldIn = OldIn;
  Param->OldZ = OldZ;
  if (channels == 2) {
    Param->OldOutR = OldOutR;
    Param->OldInR = OldInR;
    Param->OldZR = OldZR;
  }
}

void Open_PDM_Filter_64(uint8_t* data, int16_t* dataOut, uint16_t volume, TPDMFilter_InitStruct *Param)
{
  open_pdm_filter_64_common(data, dataOut, volume, Param, 16);
}

void Open_PDM_Filter_128(uint8_t* data, int16_t* dataOut, uint16_t volume, TPDMFilter_InitStruct *Param)
{
  open_pdm_filter_128_common(data, dataOut, volume, Param, 16);
}

void Open_PDM_Filter_64_24(uint8_t* data, int32_t* dataOut, uint16_t volume, TPDMFilter_InitStruct *Param)
{
  open_pdm_filter_64_common(data, dataOut, volume, Param, 24);
}

void Open_PDM_Filter_128_24(uint8_t* data, int32_t* dataOut, uint16_t volume, TPDMFilter_InitStruct *Param)
{
  open_pdm_filter_128_common(data, dataOut, volume, Param, 24);
}
