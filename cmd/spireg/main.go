// Copyright 2026 The Marusya-Gemini Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build linux

// Debug tool: dumps the meson-spicc registers of spi@13000 through /dev/mem,
// decoding the enhanced clock divider that sets the SPI rate.
package main

import (
	"fmt"
	"log"
	"os"
	"syscall"
	"unsafe"
)

const base = 0xffd13000

func main() {
	f, err := os.OpenFile("/dev/mem", os.O_RDONLY, 0)
	if err != nil {
		log.Fatalf("open /dev/mem: %v", err)
	}
	defer f.Close()
	buf := make([]byte, 0x40)
	if _, err := f.ReadAt(buf, base); err != nil {
		log.Fatalf("pread: %v", err)
	}
	rd := func(off int) uint32 { return *(*uint32)(unsafe.Pointer(&buf[off])) }
	fmt.Printf("CONREG    0x%02x = %#08x\n", 0x00, rd(0x00))
	fmt.Printf("PERIODREG 0x%02x = %#08x\n", 0x0c, rd(0x0c))
	fmt.Printf("TESTREG   0x%02x = %#08x\n", 0x30, rd(0x30))
	enh := rd(0x38)
	fmt.Printf("ENH_CTL0  0x38 = %#08x\n", enh)
	d := (enh >> 16) & 0xff
	fmt.Printf("  datarate field = %d → divider %d → %.3f MHz (parent 83.333 MHz)\n",
		d, d+1, 83.333252/float64(d+1))
	fmt.Printf("  datarate_en = %v\n", enh&(1<<24) != 0)
	_ = syscall.SYS_PREAD64
}
