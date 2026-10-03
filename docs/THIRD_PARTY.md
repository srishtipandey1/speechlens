# Third-Party Notices

## Plotly.js

The offline browser bundle at `frontend/vendor/plotly.min.js` is Plotly.js,
distributed under the MIT License. It was copied from the installed Plotly
Python package bundle; its embedded header retains the Plotly.js version and
copyright notice. The bundle is loaded locally so feature timelines and score
profiles work without a CDN connection.

MIT License

Copyright (c) 2012-2025 Plotly, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

WaveSurfer.js is loaded from cdnjs at the pinned version in
`frontend/index.html`; if it is unavailable, playback falls back to the native
HTML audio control.