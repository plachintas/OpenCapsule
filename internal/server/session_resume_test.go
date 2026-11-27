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

package server

import (
	"context"
	"log/slog"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yawaflua/marusya-gemini/internal/models"
	"github.com/yawaflua/marusya-gemini/internal/provider"
	"github.com/yawaflua/marusya-gemini/internal/transport"
)

// resumeProvider hands out sessions that answer a turn at once, and remembers
// the resumption handle each connection was opened with.
type resumeProvider struct {
	sessions chan *slowSession

	mu      sync.Mutex
	handles []string
}

func (p *resumeProvider) Name() string                     { return "resume" }
func (p *resumeProvider) OutputFormat() models.AudioFormat { return models.DefaultOutputFormat }

func (p *resumeProvider) Connect(_ context.Context, opts provider.SessionOptions) (provider.Session, error) {
	p.mu.Lock()
	p.handles = append(p.handles, opts.ResumeHandle)
	p.mu.Unlock()
	s := &slowSession{events: make(chan provider.Event, 8), reply: true}
	p.sessions <- s
	return s, nil
}

// waitHandle waits for the n-th connection and returns the handle it used.
func (p *resumeProvider) waitHandle(t *testing.T, n int) string {
	t.Helper()
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); time.Sleep(10 * time.Millisecond) {
		p.mu.Lock()
		if len(p.handles) >= n {
			h := p.handles[n-1]
			p.mu.Unlock()
			return h
		}
		p.mu.Unlock()
	}
	t.Fatalf("the %d-th provider connection did not happen", n)
	return ""
}

// resumeEnv connects a device to a server with a short idle timeout and
// returns everything the test needs.
func resumeEnv(t *testing.T, resume bool) (*resumeProvider, *transport.Conn, context.Context) {
	t.Helper()
	p := &resumeProvider{sessions: make(chan *slowSession, 8)}
	srv := httptest.NewServer(New(Config{
		Provider:    p,
		Logger:      slog.New(slog.DiscardHandler),
		IdleTimeout: 300 * time.Millisecond,
		Resume:      resume,
	}))
	t.Cleanup(srv.Close)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	t.Cleanup(cancel)
	conn, err := transport.Dial(ctx, "ws"+strings.TrimPrefix(srv.URL, "http")+transport.Path, "")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close("done") })

	conn.Send(ctx, &models.Message{Type: models.TypeHello, Hello: &models.Hello{
		ProtocolVersion: models.ProtocolVersion, DeviceID: "d", InputFormat: models.DeviceInputFormat}})
	conn.Read(ctx) // ready

	return p, conn, ctx
}

// TestResumeHandleSurvivesIdleClose keeps the handle across the idle close:
// that is the whole point - the user comes back after a pause and the
// conversation continues instead of starting over.
func TestResumeHandleSurvivesIdleClose(t *testing.T) {
	p, conn, ctx := resumeEnv(t, true)

	ps := <-p.sessions // the session warmed up on connect
	ps.events <- provider.Event{Kind: provider.EventResumeHandle, Handle: "h1", Resumable: true}
	time.Sleep(50 * time.Millisecond) // let the pump store it

	// A turn means the session was in use, so after the idle close it is
	// worth replacing.
	conn.Send(ctx, &models.Message{Type: models.TypeTurnStart, TurnID: "t1"})
	conn.SendAudio(ctx, make([]byte, 640))
	conn.Send(ctx, &models.Message{Type: models.TypeTurnEnd, TurnID: "t1"})

	if got := p.waitHandle(t, 2); got != "h1" {
		t.Fatalf("the warmed session after the idle close resumed with %q, want h1", got)
	}
}

// TestResumeHandleSurvivesBargeIn: a turn cancelled because the model was slow
// or the user interrupted must not cost the conversation - being interrupted is
// not forgetting.
func TestResumeHandleSurvivesBargeIn(t *testing.T) {
	p, conn, ctx := resumeEnv(t, true)

	ps := <-p.sessions
	ps.events <- provider.Event{Kind: provider.EventResumeHandle, Handle: "h1", Resumable: true}
	time.Sleep(50 * time.Millisecond)

	conn.Send(ctx, &models.Message{Type: models.TypeTurnStart, TurnID: "t1"})
	conn.SendAudio(ctx, make([]byte, 640))
	conn.Send(ctx, &models.Message{Type: models.TypeTurnCancel, TurnID: "t1"})

	if got := p.waitHandle(t, 2); got != "h1" {
		t.Fatalf("after a barge-in the session resumed with %q, want h1", got)
	}
}

// TestResumeDisabledStartsFresh covers the switch: with resumption off the
// server never passes a handle back.
func TestResumeDisabledStartsFresh(t *testing.T) {
	p, conn, ctx := resumeEnv(t, false)

	ps := <-p.sessions
	ps.events <- provider.Event{Kind: provider.EventResumeHandle, Handle: "h1", Resumable: true}
	time.Sleep(50 * time.Millisecond)

	conn.Send(ctx, &models.Message{Type: models.TypeTurnStart, TurnID: "t1"})
	conn.SendAudio(ctx, make([]byte, 640))
	conn.Send(ctx, &models.Message{Type: models.TypeTurnEnd, TurnID: "t1"})

	if got := p.waitHandle(t, 2); got != "" {
		t.Fatalf("resumption is off, but the session resumed with %q", got)
	}
}
