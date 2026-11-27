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
	"testing"
	"time"

	"github.com/yawaflua/marusya-gemini/internal/models"
	"github.com/yawaflua/marusya-gemini/internal/provider"
	"github.com/yawaflua/marusya-gemini/internal/transport"
)

// TestIdleClosePrewarmsOnlyAfterATurn keeps the warm-up from chasing its own
// tail: a session opened ahead of a turn, never used and closed by IdleTimeout,
// must not be replaced by another one - the replacement would idle out exactly
// the same way and cost a handshake every IdleTimeout. A session that carried a
// turn is replaced, so the next question still finds a warm one.
// replyProvider hands out sessions that answer a turn immediately, so the test
// can wait for the turn to finish and the idle timer to fire.
type replyProvider struct{ sessions chan *slowSession }

func (p *replyProvider) Name() string                     { return "reply" }
func (p *replyProvider) OutputFormat() models.AudioFormat { return models.DefaultOutputFormat }
func (p *replyProvider) Connect(context.Context, provider.SessionOptions) (provider.Session, error) {
	s := &slowSession{events: make(chan provider.Event, 8), reply: true}
	p.sessions <- s
	return s, nil
}

func TestIdleClosePrewarmsOnlyAfterATurn(t *testing.T) {
	p := &replyProvider{sessions: make(chan *slowSession, 8)}
	srv := httptest.NewServer(New(Config{
		Provider:    p,
		Logger:      slog.New(slog.DiscardHandler),
		IdleTimeout: 200 * time.Millisecond,
	}))
	defer srv.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	conn, err := transport.Dial(ctx, "ws"+strings.TrimPrefix(srv.URL, "http")+transport.Path, "")
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close("done")
	conn.Send(ctx, &models.Message{Type: models.TypeHello, Hello: &models.Hello{
		ProtocolVersion: models.ProtocolVersion, DeviceID: "d", InputFormat: models.DeviceInputFormat}})
	conn.Read(ctx) // ready

	// The connection prewarms one session.
	select {
	case <-p.sessions:
	case <-time.After(5 * time.Second):
		t.Fatal("the connection did not prewarm a session")
	}
	// It idles out and nothing replaces it.
	select {
	case <-p.sessions:
		t.Fatal("an unused session was replaced after IdleTimeout")
	case <-time.After(700 * time.Millisecond):
	}

	// A turn uses the next session; after it idles out, a warm replacement is
	// opened, because the user is evidently talking to the speaker.
	conn.Send(ctx, &models.Message{Type: models.TypeTurnStart, TurnID: "t1"})
	conn.SendAudio(ctx, make([]byte, 640))
	conn.Send(ctx, &models.Message{Type: models.TypeTurnEnd, TurnID: "t1"})
	select {
	case <-p.sessions:
	case <-time.After(5 * time.Second):
		t.Fatal("a turn did not open a session")
	}
	select {
	case <-p.sessions:
	case <-time.After(5 * time.Second):
		t.Fatal("a used session was not replaced after IdleTimeout")
	}
}
