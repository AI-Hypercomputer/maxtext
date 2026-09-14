// Loads the dashboard for a jsdom check.
//
// The published page reads its numbers from views/*.json with fetch. A check
// has no web server and no data files, so it asks the page for the sample data
// that is already inside it by setting one flag before the page's own script
// runs. Nothing sets that flag on the published page.
const fs = require('fs');
const path = require('path');

const PAGE = path.join(__dirname, '..', '..', '..', '..', 'dev', 'bench', 'ci-pulse', 'index.html');

module.exports = function pageHtml() {
  const html = fs.readFileSync(PAGE, 'utf8');
  if (!html.includes('<script>')) throw new Error('the dashboard has no <script> block');
  return html.replace('<script>', '<script>window.__CI_PULSE_MOCK__=true;</script>\n<script>');
};
module.exports.PAGE = PAGE;
