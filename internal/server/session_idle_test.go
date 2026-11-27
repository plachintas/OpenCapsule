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
	"errors"
	"fmt"
	"testing"
)

// TestIsIdleClose keeps the provider's inactivity close out of the error path:
// a prewarmed session is dropped by Google after a couple of minutes without
// audio, and the panel must not show that as a failure.
func TestIsIdleClose(t *testing.T) {
	idle := fmt.Errorf("websocket: close 1008 (policy violation): Session closed due to client inactivity.")
	if !isIdleClose(idle) {
		t.Fatalf("inactivity close must be recognised: %v", idle)
	}
	for _, err := range []error{
		nil,
		errors.New("websocket: close 1008 (policy violation): other reason"),
		errors.New("websocket: close 1006 (abnormal closure)"),
		errors.New("dial tcp 1.2.3.4:443: i/o timeout"),
	} {
		if isIdleClose(err) {
			t.Fatalf("%v must not count as an idle close", err)
		}
	}
}
