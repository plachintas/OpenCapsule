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

// Debug tool: writes one LED ring frame with the agent's parameters and can
// put the SPI controller into internal loopback (SPI_LOOP) to read back what
// the driver actually shifts out. Not part of the agent.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"log"
	"runtime"
	"syscall"
	"time"
	"unsafe"
)

const (
	resetBytes = 128
	pixels     = 12
	spiSpeed   = 3340000
	primeSpeed = 1000000
	primeBytes = 16

	spiIOCWRMode   = 0x40016b01
	spiIOCMessage2 = 0x40000000 | (2*32)<<16 | 'k'<<8
	spiIOCMessage1 = 0x40000000 | (1*32)<<16 | 'k'<<8
	spiLoop        = 0x20
)

var symbols = [4]byte{0x88, 0x8c, 0xc8, 0xcc}

func encode(f [pixels][3]uint8, reverse bool) []byte {
	n := 2*resetBytes + pixels*12
	n = (n + 7) &^ 7
	buf := make([]byte, n)
	p := buf[resetBytes:]
	for _, px := range f {
		for _, v := range [3]uint8{px[1], px[0], px[2]} { // G, R, B
			p[0] = symbols[v>>6&3]
			p[1] = symbols[v>>4&3]
			p[2] = symbols[v>>2&3]
			p[3] = symbols[v&3]
			p = p[4:]
		}
	}
	if reverse {
		for i := 0; i < len(buf); i += 8 {
			w := buf[i : i+8]
			for a, b := 0, 7; a < b; a, b = a+1, b-1 {
				w[a], w[b] = w[b], w[a]
			}
		}
	}
	return buf
}

type spiTransfer struct {
	txBuf, rxBuf   uint64
	len, speedHz   uint32
	delayUsecs     uint16
	bitsPerWord    uint8
	csChange       uint8
	txNbits        uint8
	rxNbits        uint8
	wordDelayUsecs uint8
	_              uint8
}

func main() {
	color := flag.String("color", "0040ff", "RRGGBB")
	bright := flag.Int("bright", 50, "per cent")
	loop := flag.Bool("loop", false, "enable SPI internal loopback")
	prime := flag.Bool("prime", true, "send the 16-byte primer transfer")
	bpw := flag.Uint("bpw", 64, "bits per word")
	extra := flag.Int("extra", 0, "extra zero bytes appended to the frame (reset, harmless)")
	rev := flag.Bool("rev", true, "reverse every 8-byte group (required by the 64-bit DMA path)")
	onepx := flag.Bool("onepx", false, "light only pixel 0, the rest dark")
	speed := flag.Uint("speed", spiSpeed, "SPI clock to request, Hz")
	rebind := flag.Bool("rebind", false, "unbind/bind meson-spicc before the frame")
	path := flag.String("dev", "/dev/spidev0.0", "spidev node")
	warm := flag.Uint("warm", 0, "send an 8-byte transfer at this rate in its own ioctl before the frame, Hz")
	flag.Parse()

	var c [3]uint8
	fmt.Sscanf(*color, "%02x%02x%02x", &c[0], &c[1], &c[2])
	var frame [pixels][3]uint8
	for i := range frame {
		frame[i] = [3]uint8{
			uint8(int(c[0]) * *bright / 100),
			uint8(int(c[1]) * *bright / 100),
			uint8(int(c[2]) * *bright / 100),
		}
	}
	if *onepx {
		for i := 1; i < pixels; i++ {
			frame[i] = [3]uint8{}
		}
	}
	tx := encode(frame, *rev)
	if *rebind {
		if err := doRebind(); err != nil {
			log.Fatalf("rebind: %v", err)
		}
	}
	if *extra > 0 {
		tx = append(tx, make([]byte, *extra)...)
	}

	fd, err := syscall.Open(*path, syscall.O_RDWR|syscall.O_CLOEXEC, 0)
	if err != nil {
		log.Fatalf("open: %v", err)
	}
	defer syscall.Close(fd)

	mode := byte(0)
	if *loop {
		mode |= spiLoop
	}
	if _, _, e := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), spiIOCWRMode, uintptr(unsafe.Pointer(&mode))); e != 0 {
		log.Fatalf("SPI_IOC_WR_MODE: %v", e)
	}

	if *warm > 0 {
		wb := make([]byte, 8)
		wx := spiTransfer{txBuf: uint64(uintptr(unsafe.Pointer(&wb[0]))), rxBuf: uint64(uintptr(unsafe.Pointer(&wb[0]))),
			len: 8, speedHz: uint32(*warm), bitsPerWord: uint8(*bpw)}
		if _, _, e := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), uintptr(spiIOCMessage1), uintptr(unsafe.Pointer(&wx))); e != 0 {
			log.Fatalf("warm transfer: %v", e)
		}
		runtime.KeepAlive(wb)
		time.Sleep(2 * time.Millisecond)
		log.Printf("warmed the divider at %d Hz in a separate transfer", *warm)
	}

	rx := make([]byte, len(tx))
	primeBuf := make([]byte, primeBytes)
	xfer := [2]spiTransfer{
		{
			txBuf: uint64(uintptr(unsafe.Pointer(&primeBuf[0]))), rxBuf: uint64(uintptr(unsafe.Pointer(&primeBuf[0]))),
			len: primeBytes, speedHz: primeSpeed, bitsPerWord: uint8(*bpw),
		},
		{
			txBuf: uint64(uintptr(unsafe.Pointer(&tx[0]))), rxBuf: uint64(uintptr(unsafe.Pointer(&rx[0]))),
			len: uint32(len(tx)), speedHz: uint32(*speed), bitsPerWord: uint8(*bpw),
		},
	}
	code := uintptr(spiIOCMessage2)
	if !*prime {
		code = spiIOCMessage1
	}
	start := time.Now()
	_, _, e := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), code, uintptr(unsafe.Pointer(&xfer[0])))
	elapsed := time.Since(start)
	if e != 0 {
		log.Fatalf("SPI_IOC_MESSAGE: %v", e)
	}
	bits := float64(len(tx)) * 8
	if *prime {
		bits += float64(primeBytes) * 8
	}
	log.Printf("ioctl took %v for %.0f bits → %.0f bit/s (frame %d bytes)", elapsed, bits, bits/elapsed.Seconds(), len(tx))
	log.Printf("sent %d bytes, mode=%#x loop=%v bpw=%d prime=%v", len(tx), mode, *loop, *bpw, *prime)

	if *loop {
		if bytes.Equal(tx, rx) {
			log.Printf("loopback: rx == tx (%d bytes) — controller shifts our buffer unchanged", len(rx))
		} else {
			diff := 0
			for i := range tx {
				if tx[i] != rx[i] {
					diff++
				}
			}
			log.Printf("loopback: %d of %d bytes differ", diff, len(tx))
			for i := 0; i < 48 && i < len(tx); i += 8 {
				log.Printf("  %3d tx % x  rx % x", i, tx[i:i+8], rx[i:i+8])
			}
		}
	}
}

// rebind unbinds and binds the meson-spicc controller, then waits for the
// spidev node to reappear.
func doRebind() error {
	const dir = "/sys/bus/platform/drivers/meson-spicc"
	for _, name := range []string{"unbind", "bind"} {
		fd, err := syscall.Open(dir+"/"+name, syscall.O_WRONLY, 0)
		if err != nil {
			return err
		}
		if _, err := syscall.Write(fd, []byte("ffd13000.spi")); err != nil {
			syscall.Close(fd)
			return err
		}
		syscall.Close(fd)
	}
	deadline := time.Now().Add(2 * time.Second)
	for {
		if err := syscall.Access("/dev/spidev0.0", 2); err == nil {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("spidev node did not come back")
		}
		time.Sleep(time.Millisecond)
	}
}
