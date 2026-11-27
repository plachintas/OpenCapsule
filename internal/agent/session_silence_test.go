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

package agent_test

import (
	"context"
	"log/slog"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/yawaflua/marusya-gemini/internal/agent"
	"github.com/yawaflua/marusya-gemini/internal/controller"
	"github.com/yawaflua/marusya-gemini/internal/indicator"
	"github.com/yawaflua/marusya-gemini/internal/provider/echo"
	"github.com/yawaflua/marusya-gemini/internal/server"
	"github.com/yawaflua/marusya-gemini/internal/transport"
	"github.com/yawaflua/marusya-gemini/internal/wakeword"
)

// TestShortBlipAfterWakeStartsNoTurn covers the case the user complained about:
// the wake word is heard, the room is silent, and the speaker used to send the
// silence to the model, wait eight seconds for an answer and give up. A blip
// shorter than MinUtterance must end the listening window right away, without
// the model ever being woken.
func TestShortBlipAfterWakeStartsNoTurn(t *testing.T) {
	log := slog.New(slog.NewTextHandler(newTestWriter(t), &slog.HandlerOptions{Level: slog.LevelDebug}))
	srv := httptest.NewServer(server.New(server.Config{Provider: echo.Provider{}, AuthToken: "secret", Logger: log}))
	defer srv.Close()

	mic := &fakeMic{}
	player := &fakePlayer{done: make(chan struct{})}
	wake := make(fakeWake, 1)
	ind := &fakeIndicator{}
	a := agent.New(agent.Config{
		DeviceID:     "test",
		ServerURL:    "ws" + strings.TrimPrefix(srv.URL, "http") + transport.Path,
		Token:        "secret",
		MinUtterance: 300 * time.Millisecond,
	}, log, agent.Deps{Capture: mic, Player: player, Detector: wake, Dispatcher: &controller.Dispatcher{}, Indicator: ind})

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	stopped := make(chan struct{})
	go func() {
		a.Run(ctx)
		close(stopped)
	}()
	defer func() {
		cancel()
		<-stopped
	}()

	time.Sleep(300 * time.Millisecond) // connect and learn the noise floor
	wake <- wakeword.Detection{Source: "test"}
	time.Sleep(20 * time.Millisecond)
	mic.speechFrames.Store(8) // 160 ms: a click, not a request
	time.Sleep(500 * time.Millisecond)

	if got := ind.get(); slices.Contains(got, indicator.Thinking) {
		t.Fatalf("indicator states = %v, the model was woken for a blip", got)
	}
	select {
	case <-player.done:
		t.Fatal("a reply was played for an utterance that never was one")
	default:
	}
}
