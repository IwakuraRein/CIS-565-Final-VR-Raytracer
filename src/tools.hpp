/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#ifndef TOOLS_H
#define TOOLS_H

// Utility to time the execution of something resetting the timer
// on each elapse call
// Usage:
// {
//   MilliTimer timer;
//   ... stuff ...
//   double time_elapse = timer.elapse();
// }
#include <chrono>
#include <sstream>
#include <ios>

#include "nvh/nvprint.hpp"
#include "nvh/timesampler.hpp"

struct MilliTimer : public nvh::Stopwatch
{
  void print() { LOGI(" --> (%5.3f ms)\n", elapsed()); }
};

#define gigo 1000000000.0f
#define mego 1000000.0f

// Formating with local number representation
template <class T>
std::string FormatNumbers(T value)
{
  std::stringstream ss;
  ss.imbue(std::locale(""));
  //tigra: only integer - format it to kilo, mega, giga
  
  if(value > 10005) {	  
			float num, conv_num;
			char kb_mega_giga, str[30];
			
			num = (float) value;
			
			conv_num = num;
			kb_mega_giga = '\0';
			if(num >= gigo)
			{
				kb_mega_giga = 'G';
				conv_num = float(num) / gigo;
			}
			else
			if(num >= mego)
			{
				kb_mega_giga = 'm';
				conv_num = float(num) / mego;
			}
			else
			if(num >= 1000.0f)
			{
				kb_mega_giga = 'k';
				conv_num = float(num) / 1000.0f;
			}
			
			sprintf(str, "%.2f%c", conv_num, kb_mega_giga);
	
			/*
			ss << str << " (";			
			ss << std::fixed << value << ")";
			*/
			
			return str;
  } else {	  
	ss << std::fixed << value;
  }
  
  return ss.str();
}

template<typename T>
inline float luminance(const T& color)
{
	return color[0] * 0.2126f + color[1] * 0.7152f + color[2] * 0.0722f;
}

#endif
