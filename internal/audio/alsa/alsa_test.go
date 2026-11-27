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

package alsa

import (
	"context"
	"io"
	"log/slog"
	"os"
	"sync"
	"testing"

	"github.com/yawaflua/marusya-gemini/internal/models"
)

// fakeMuter records the mute calls the player makes.
type fakeMuter struct {
	mu    sync.Mutex
	calls []bool
}

func (f *fakeMuter) SetMuted(_ context.Context, muted bool) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, muted)
	return nil
}

func (f *fakeMuter) recorded() []bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]bool(nil), f.calls...)
}

// truePath finds a binary that ignores its arguments and exits at once.
func truePath(t *testing.T) string {
	for _, p := range []string{"/bin/true", "/usr/bin/true"} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	t.Skip("no true binary")
	return ""
}

// TestPlayerMutesWhileIdle checks the whole point of the setting: the mixer
// control is unmuted before playback starts and muted again once the player
// exits, so the amplifier does not hiss on digital silence.
func TestPlayerMutesWhileIdle(t *testing.T) {
	mute := &fakeMuter{}
	// /bin/true ignores the aplay arguments and exits at once.
	p := NewPlayer(Config{
		AplayPath: truePath(t),
		Mute:      mute,
		Logger:    slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	st, err := p.Open(context.Background(), models.AudioFormat{SampleRateHertz: 24000, Channels: 1})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if got := mute.recorded(); len(got) != 1 || got[0] {
		t.Fatalf("on open the control must be unmuted, got %v", got)
	}
	if err := st.Drain(); err != nil {
		t.Logf("drain: %v", err) // writing to a finished aplay may fail; the mute still has to happen
	}
	got := mute.recorded()
	if len(got) < 2 || !got[len(got)-1] {
		t.Fatalf("after playback the control must be muted, got %v", got)
	}
}

// TestPlayerWithoutMute keeps the default path free of mixer work.
func TestPlayerWithoutMute(t *testing.T) {
	p := NewPlayer(Config{AplayPath: truePath(t), Logger: slog.New(slog.NewTextHandler(io.Discard, nil))})
	st, err := p.Open(context.Background(), models.AudioFormat{SampleRateHertz: 24000, Channels: 1})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if err := st.Drain(); err != nil {
		t.Logf("drain: %v", err)
	}
}
