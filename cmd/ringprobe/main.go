// Copyright 2026 The Marusya-Gemini Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Command ringprobe writes plain frames to the LED ring with the agent's own
// strip code, for bisecting the SPI transport on a real speaker: one frame
// like a static look, or a run of frames like an animation.
package main

import (
	"flag"
	"log"
	"math"
	"time"

	"github.com/yawaflua/marusya-gemini/internal/indicator/ringled"
)

func main() {
	color := flag.String("color", "0040ff", "RRGGBB")
	bright := flag.Int("bright", 50, "brightness in per cent")
	frames := flag.Int("frames", 1, "how many frames to write")
	gap := flag.Duration("gap", 0, "pause between frames")
	flag.Parse()

	c, err := ringled.ParseRGB(*color)
	if err != nil {
		log.Fatal(err)
	}
	m := float64(*bright) / 100
	q := func(v uint8) uint8 { return uint8(math.Round(math.Min(float64(v)*m, 255))) }
	var f ringled.Frame
	for i := range f {
		f[i] = ringled.RGB{R: q(c.R), G: q(c.G), B: q(c.B)}
	}
	s := ringled.NewSPI(ringled.DefaultDevice)
	start := time.Now()
	for i := 0; i < *frames; i++ {
		if err := s.Write(f); err != nil {
			log.Fatalf("frame %d: %v", i, err)
		}
		if *gap > 0 && i != *frames-1 {
			time.Sleep(*gap)
		}
	}
	log.Printf("wrote %d frame(s) in %v", *frames, time.Since(start))
}
