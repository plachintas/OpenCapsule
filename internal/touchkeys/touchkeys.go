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

// Package touchkeys handles the speaker's touch panel (Hynitron CSK05T):
//
//	VOL+ / VOL-     volume, auto-repeat while held, level shown on the display
//	centre, short   activate the assistant
//	centre, long    Bluetooth pairing (bt-pair)
//	PROG1           next track
//	MICMUTE         mute the microphone
//
// With [Config.MediaKeys], while music plays (see [models.MediaStatus]):
//
//	VOL+ / VOL-, double tap   next / previous track; the volume step of
//	                          the first tap is undone
//	centre, short             pause; pressed again within
//	                          [Config.ResumeWindow], play on
//
// A single press and holding change the volume as always, so the volume
// keys never lose their main job.
//
// [Panel] is a [wakeword.Detector], so the centre key activates the agent
// exactly like a wake word.
package touchkeys

import (
	"context"
	"log/slog"
	"strconv"
	"time"

	"github.com/yawaflua/marusya-gemini/internal/controller"
	"github.com/yawaflua/marusya-gemini/internal/evdev"
	"github.com/yawaflua/marusya-gemini/internal/models"
	"github.com/yawaflua/marusya-gemini/internal/wakeword"
)

// DefaultDeviceName is the input device name of the touch controller.
const DefaultDeviceName = "Hynitron CSK05T"

// Linux key codes used by the panel.
const (
	KeyVolumeDown = 114
	KeyVolumeUp   = 115
	KeyProg1      = 148
	KeyMicMute    = 248
	KeyVoice      = 582
)

// Config configures a [Panel].
type Config struct {
	// Device is the evdev node; empty looks it up by Name.
	Device string
	Name   string

	CenterKey, NextKey, MicKey, VolumeUpKey, VolumeDownKey uint16

	// VolumeStep is the change per press, per cent.
	VolumeStep int
	// LongPress is how long the centre key must be held for pairing.
	LongPress time.Duration
	// RepeatDelay and RepeatInterval control volume auto-repeat.
	RepeatDelay, RepeatInterval time.Duration
	// PairTimeout is passed to [controller.Bluetooth.Pair].
	PairTimeout time.Duration
	// MediaKeys makes the keys control the track while music plays: a
	// double tap on a volume key skips the track, the centre key pauses
	// and resumes.
	MediaKeys bool
	// DoubleTap is the longest gap between the release of a volume key
	// and the next press of it that makes a double tap.
	DoubleTap time.Duration
	// ResumeWindow is how long after a pause by the centre key a short
	// press plays on instead of activating the assistant.
	ResumeWindow time.Duration
}

// DefaultConfig returns the VK Capsula Mini layout.
func DefaultConfig() Config {
	return Config{
		Name:           DefaultDeviceName,
		CenterKey:      KeyVoice,
		NextKey:        KeyProg1,
		MicKey:         KeyMicMute,
		VolumeUpKey:    KeyVolumeUp,
		VolumeDownKey:  KeyVolumeDown,
		VolumeStep:     5,
		LongPress:      1500 * time.Millisecond,
		RepeatDelay:    400 * time.Millisecond,
		RepeatInterval: 150 * time.Millisecond,
		PairTimeout:    2 * time.Minute,
		MediaKeys:      true,
		DoubleTap:      350 * time.Millisecond,
		ResumeWindow:   10 * time.Minute,
	}
}

// KeyName returns the name of a key of the panel: "voice", "next", "mic",
// "volume_up", "volume_down", or "key_<code>" for any other key.
func (c Config) KeyName(code uint16) string {
	switch code {
	case c.CenterKey:
		return "voice"
	case c.NextKey:
		return "next"
	case c.MicKey:
		return "mic"
	case c.VolumeUpKey:
		return "volume_up"
	case c.VolumeDownKey:
		return "volume_down"
	}
	return "key_" + strconv.Itoa(int(code))
}

// Overlayer shows short feedback, e.g. the volume level.
type Overlayer interface {
	Overlay(text string, hold time.Duration)
}

type nopOverlay struct{}

func (nopOverlay) Overlay(string, time.Duration) {}

// Panel reads the touch keys and performs their actions.
type Panel struct {
	Config     Config
	Dispatcher *controller.Dispatcher
	// Display shows feedback; optional.
	Display Overlayer
	// OnMicMute is called when the microphone is muted or unmuted.
	OnMicMute func(muted bool)
	// OnKey, if set, is called for every press and release with the key
	// name (see [Config.KeyName]), before the key's own action.
	OnKey func(name string, code uint16, pressed bool)
	// Busy, if set, reports that the assistant is listening or answering:
	// the centre key then always goes to the assistant (to stop or barge
	// in), not to the music.
	Busy   func() bool
	Logger *slog.Logger

	muted    bool
	pausedAt time.Time // when the centre key paused the music
}

var _ wakeword.Detector = (*Panel)(nil)

// WantsAudio implements [wakeword.Detector].
func (p *Panel) WantsAudio() bool { return false }

// Run implements [wakeword.Detector].
func (p *Panel) Run(ctx context.Context, _ <-chan []byte, out chan<- wakeword.Detection) error {
	if p.Display == nil {
		p.Display = nopOverlay{}
	}
	if p.Logger == nil {
		p.Logger = slog.Default()
	}
	dev := p.Config.Device
	if dev == "" {
		var err error
		if dev, err = evdev.FindByName(p.Config.Name); err != nil {
			return err
		}
	}
	p.Logger.Info("touch keys enabled", "device", dev)

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	events := make(chan evdev.KeyEvent, 16)
	errc := make(chan error, 1)
	go func() { errc <- evdev.ReadKeys(ctx, dev, events) }()

	var (
		longT   *time.Timer
		longC   <-chan time.Time
		longHit bool
		repT    *time.Timer
		repC    <-chan time.Time
		repD    int
		// repeated: the volume key being held started auto-repeat.
		repeated bool
		// pressLevel is the volume before the current volume press.
		pressLevel int
		pressOK    bool
		// lastTap is a short volume press, the first half of a double tap.
		lastTap tap
	)
	stopRepeat := func() {
		if repT != nil {
			repT.Stop()
		}
		repC, repD, repeated = nil, 0, false
	}
	for {
		select {
		case <-ctx.Done():
			return nil
		case err := <-errc:
			return err

		case <-repC:
			repeated = true
			p.adjustVolume(ctx, repD)
			repT = time.NewTimer(p.Config.RepeatInterval)
			repC = repT.C

		case <-longC:
			longC, longHit = nil, true
			p.pair()

		case ev := <-events:
			if ev.Value == evdev.KeyRepeat {
				continue // the kernel's repeat is replaced by ours
			}
			pressed := ev.Value == evdev.KeyPress
			if p.OnKey != nil {
				p.OnKey(p.Config.KeyName(ev.Code), ev.Code, pressed)
			}
			switch ev.Code {
			case p.Config.VolumeUpKey, p.Config.VolumeDownKey:
				if !pressed {
					// A short press may start a double tap.
					lastTap = tap{}
					if repC != nil && !repeated && pressOK {
						lastTap = tap{code: ev.Code, at: time.Now(), level: pressLevel}
					}
					stopRepeat()
					continue
				}
				stopRepeat()
				if p.doubleTap(ctx, lastTap, ev.Code) {
					lastTap = tap{}
					continue // no volume change and no repeat for this press
				}
				lastTap = tap{}
				repD = p.Config.VolumeStep
				if ev.Code == p.Config.VolumeDownKey {
					repD = -repD
				}
				pressLevel, pressOK = p.adjustVolume(ctx, repD)
				repT = time.NewTimer(p.Config.RepeatDelay)
				repC = repT.C

			case p.Config.CenterKey:
				if pressed {
					longHit = false
					longT = time.NewTimer(p.Config.LongPress)
					longC = longT.C
				} else if longT != nil {
					longT.Stop()
					longC = nil
					if !longHit && !p.pauseOrResume(ctx) {
						p.activate(out)
					}
				}

			case p.Config.NextKey:
				if pressed {
					p.next(ctx)
				}

			case p.Config.MicKey:
				if pressed {
					p.toggleMic()
				}
			}
		}
	}
}

func (p *Panel) activate(out chan<- wakeword.Detection) {
	if p.muted {
		p.Display.Overlay("OFF", 800*time.Millisecond)
		return
	}
	select {
	case out <- wakeword.Detection{Source: "touch", At: time.Now()}:
	default:
	}
}

// adjustVolume changes the volume by delta and reports the level before.
func (p *Panel) adjustVolume(ctx context.Context, delta int) (before int, ok bool) {
	v := p.Dispatcher.Volume
	if v == nil {
		return 0, false
	}
	st, err := v.Get(ctx)
	if err == nil {
		if err = p.setVolume(ctx, st.Level+delta); err == nil {
			return st.Level, true
		}
	}
	p.Logger.Warn("touch keys: volume", "err", err)
	p.Display.Overlay("Err", time.Second)
	return 0, false
}

func (p *Panel) setVolume(ctx context.Context, level int) error {
	level = max(0, min(100, level))
	if err := p.Dispatcher.Volume.Set(ctx, level); err != nil {
		return err
	}
	p.Display.Overlay(strconv.Itoa(level), 1500*time.Millisecond)
	return nil
}

// tap is a short press of a volume key.
type tap struct {
	code  uint16
	at    time.Time // release
	level int       // volume before the press
}

// doubleTap handles the second press of a double tap on a volume key with
// [Config.MediaKeys] while music plays: it undoes the volume step of the
// first tap and skips the track. It reports whether the press was used.
func (p *Panel) doubleTap(ctx context.Context, last tap, code uint16) bool {
	if !p.Config.MediaKeys || last.code != code || time.Since(last.at) > p.Config.DoubleTap ||
		!p.mediaStatus(ctx).Playing {
		return false
	}
	if err := p.setVolume(ctx, last.level); err != nil {
		p.Logger.Warn("touch keys: volume", "err", err)
	}
	p.skip(ctx, code == p.Config.VolumeUpKey)
	return true
}

func (p *Panel) pair() {
	bt := p.Dispatcher.Bluetooth
	if bt == nil {
		return
	}
	p.Display.Overlay("bt", 2*time.Second)
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := bt.Pair(ctx, p.Config.PairTimeout); err != nil {
			p.Logger.Warn("touch keys: bluetooth pairing", "err", err)
			p.Display.Overlay("Err", 1500*time.Millisecond)
		}
	}()
}

func (p *Panel) next(ctx context.Context) {
	m := p.Dispatcher.Media
	if m == nil {
		return
	}
	if err := m.Control(ctx, models.MediaNext); err != nil {
		p.Logger.Warn("touch keys: next track", "err", err)
	}
}

// mediaStatus reports what the media player does; nothing on errors.
func (p *Panel) mediaStatus(ctx context.Context) models.MediaStatus {
	m := p.Dispatcher.Media
	if m == nil {
		return models.MediaStatus{}
	}
	ctx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	st, err := m.Status(ctx)
	if err != nil {
		p.Logger.Debug("touch keys: media status", "err", err)
		return models.MediaStatus{}
	}
	return st
}

// pauseOrResume handles a short centre press with [Config.MediaKeys]: it
// pauses the music playing, or plays on what this key paused a moment ago.
// It reports whether the press was used; otherwise the assistant starts.
func (p *Panel) pauseOrResume(ctx context.Context) bool {
	if !p.Config.MediaKeys || p.Dispatcher.Media == nil || (p.Busy != nil && p.Busy()) {
		return false
	}
	st := p.mediaStatus(ctx)
	var op models.MediaOperation
	switch {
	case st.Playing:
		op = models.MediaPause
	case st.Active() && !p.pausedAt.IsZero() && time.Since(p.pausedAt) < p.Config.ResumeWindow:
		op = models.MediaPlay
	default:
		p.pausedAt = time.Time{}
		return false
	}
	if err := p.Dispatcher.Media.Control(ctx, op); err != nil {
		p.Logger.Warn("touch keys: "+string(op), "err", err)
		return true
	}
	if op == models.MediaPause {
		p.pausedAt = time.Now()
	} else {
		p.pausedAt = time.Time{}
	}
	return true
}

// skip goes to the next or the previous track.
func (p *Panel) skip(ctx context.Context, next bool) {
	op := models.MediaPrevious
	if next {
		op = models.MediaNext
	}
	if err := p.Dispatcher.Media.Control(ctx, op); err != nil {
		p.Logger.Warn("touch keys: "+string(op), "err", err)
	}
}

func (p *Panel) toggleMic() {
	p.muted = !p.muted
	// No text on the display: the LED ring and the face show the state.
	if p.OnMicMute != nil {
		p.OnMicMute(p.muted)
	}
}
