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

package gemini

import (
	"testing"

	"google.golang.org/genai"

	"github.com/yawaflua/marusya-gemini/internal/provider"
)

// TestLiveConfigSessionResumption pins the two halves of the resumption
// handshake: a handle given by the server is passed to the backend, and with no
// handle the session still asks for updates, otherwise there would never be one
// to continue with.
func TestLiveConfigSessionResumption(t *testing.T) {
	p := &Provider{}
	if got := p.liveConfig(provider.SessionOptions{ResumeHandle: "h1"}, ""); got.SessionResumption == nil ||
		got.SessionResumption.Handle != "h1" {
		t.Fatalf("resume handle not passed: %+v", got.SessionResumption)
	}
	got := p.liveConfig(provider.SessionOptions{}, "")
	if got.SessionResumption == nil || got.SessionResumption.Handle != "" {
		t.Fatalf("a fresh session must ask for handles: %+v", got.SessionResumption)
	}
}

// TestTranslateResumeHandle maps the backend message onto the event the server
// keeps the handle from.
func TestTranslateResumeHandle(t *testing.T) {
	events := translate(&genai.LiveServerMessage{SessionResumptionUpdate: &genai.LiveServerSessionResumptionUpdate{
		NewHandle: "h2", Resumable: true,
	}})
	if len(events) != 1 || events[0].Kind != provider.EventResumeHandle ||
		events[0].Handle != "h2" || !events[0].Resumable {
		t.Fatalf("unexpected events: %+v", events)
	}

	// At points the backend cannot resume (a tool call in flight) it sends
	// no handle and resumable=false; the event still arrives so the server
	// can tell the difference.
	events = translate(&genai.LiveServerMessage{SessionResumptionUpdate: &genai.LiveServerSessionResumptionUpdate{
		Resumable: false,
	}})
	if len(events) != 1 || events[0].Kind != provider.EventResumeHandle || events[0].Resumable {
		t.Fatalf("unexpected events: %+v", events)
	}
}
