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

package audio

import (
	"slices"

	"github.com/yawaflua/marusya-gemini/internal/models"
)

// Earcon is a short non-speech sound signalling a state change.
type Earcon int

const (
	// EarconListen is played when the assistant starts listening.
	EarconListen Earcon = iota
	// EarconDone is played when listening stops without a request.
	EarconDone
	// EarconError is played when the request failed.
	EarconError
	// EarconNotify precedes a message the server sends on its own.
	EarconNotify
	// EarconMute is played when the microphone is muted: a falling pair, so
	// the key feels like "off" without a word.
	EarconMute
	// EarconUnmute is played when the microphone is unmuted: the mirror pair.
	EarconUnmute
	// EarconVolume is played when the volume keys change the level. One sound
	// for both directions: the level itself is on the display.
	EarconVolume
)

// Render synthesizes the earcon in format f.
func (e Earcon) Render(f models.AudioFormat) []byte {
	const g = 0.25
	switch e {
	case EarconListen:
		return slices.Concat(Tone(f, 660, 70, g), Silence(f, 20), Tone(f, 990, 90, g))
	case EarconDone:
		return slices.Concat(Tone(f, 990, 70, g), Silence(f, 20), Tone(f, 660, 90, g))
	case EarconNotify:
		return slices.Concat(Tone(f, 784, 90, g), Silence(f, 30), Tone(f, 988, 90, g), Silence(f, 30), Tone(f, 1175, 140, g))
	case EarconMute:
		return slices.Concat(Tone(f, 494, 90, g), Silence(f, 20), Tone(f, 330, 130, g))
	case EarconUnmute:
		return slices.Concat(Tone(f, 587, 90, g), Silence(f, 20), Tone(f, 880, 130, g))
	case EarconVolume:
		return slices.Concat(Tone(f, 1046, 45, g), Silence(f, 40), Tone(f, 1046, 45, g))
	default:
		return slices.Concat(Tone(f, 330, 150, g), Silence(f, 40), Tone(f, 247, 220, g))
	}
}
