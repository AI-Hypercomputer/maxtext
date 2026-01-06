/**
 * Handles inline editable commands in documentation.
 * Replaces placeholders in code blocks with inline input fields.
 */
document.addEventListener('DOMContentLoaded', () => {
  const codeBlocks = document.querySelectorAll('div.highlight-sh pre, div.highlight-bash pre, div.highlight-default pre');

  codeBlocks.forEach(block => {

    const originalHTML = block.innerHTML;

    const placeholders = [
      "<your virtual env name>",
      "<model name>",
      "<tokenizer path>",
      "<Hugging Face access token>",
      "<output directory to store run logs>",
      "<name for this run>",
      "<number of fine-tuning steps to run>",
      "<batch size per device>",
      "<Hugging Face dataset name>",
      "<data split for train>",
      "<data columns to train on>",
      "<gcs path for MaxText checkpoint>",
      "<Google Cloud Project ID>",
      "<Name of GKE Cluster>",
      "<GKE Cluster Zone>",
      "<Name of Workload>",
      "<TPU Type>",
      "<GCS Path for Output/Logs>",
      "<Fine-Tuning Steps>",
      "<Hugging Face Access Token>",
      "<Model Name>",
      "<Model Tokenizer>",
      "<Hugging Face Dataset Name>",
      "<Data Split for Train>",
      "<Data Columns to Train on>",
      "<cluster name>",
      "<GCP project ID>",
      "<zone name>",
      "<path/to/gcr.io>",
      "<number of slices>",
      "<Flag to use zarr3>",
      "<Flag to use ocdbt>",
      "<Hugging Face Model>",
      "<MaxText Model>",
      "<Tokenizer>",
      "<Name for this run>",
      "<Docker Image Name>"
    ];

    let newHTML = originalHTML;

    placeholders.forEach(placeholder => {
      // 1. create robust regex for this placeholder
      // escape chars
      const escapeRegex = (string) => string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

      const htmlEscapedKey = placeholder
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

      let pattern = '';
      for (let i = 0; i < htmlEscapedKey.length; i++) {
        const char = htmlEscapedKey[i];
        pattern += escapeRegex(char) + '(?:<[^>]+>)*';
      }

      const regex = new RegExp(pattern, 'g');

      // Replace with an input element
      // We use the original placeholder text as placeholder for the input
      const inputHTML = `<input class="inline-input" placeholder="${placeholder}" style="width: ${placeholder.length + 2}ch;" />`;

      newHTML = newHTML.replace(regex, inputHTML);
    });

    if (newHTML !== originalHTML) {
      block.innerHTML = newHTML;
    }
  });

  // Add event listeners to newly created inputs to auto-resize
  document.querySelectorAll('.inline-input').forEach(input => {
    input.addEventListener('input', function () {
      this.style.width = Math.max(this.value.length, this.placeholder.length) + 2 + 'ch';
    });
  });

  /**
   * Intercept copy button clicks to include user input values.
   * Runs in capture phase to precede sphinx-copybutton's listener.
   */
  document.addEventListener('click', (event) => {
    // Check if the clicked element is a copy button or inside one
    const button = event.target.closest('.copybtn');
    if (!button) return;

    // Find the associated code block
    // Sphinx-copybutton places the button inside .highlight usually
    const highlightDiv = button.closest('.highlight');
    if (!highlightDiv) return;

    const inputs = highlightDiv.querySelectorAll('input.inline-input');
    if (inputs.length === 0) return;

    const swaps = [];
    inputs.forEach(input => {
      // Create a temporary span with the input's current value
      const span = document.createElement('span');
      // If value is empty, fallback to placeholder to match original text behavior
      const val = input.value;
      span.textContent = val ? val : input.placeholder;

      // Mimic input appearance slightly if needed, but plain text is what we want copied
      span.style.color = val ? 'inherit' : 'gray';

      input.replaceWith(span);
      swaps.push({ input, span });
    });

    // Revert immediately after the current event loop
    setTimeout(() => {
      swaps.forEach(({ input, span }) => {
        span.replaceWith(input);
      });
    }, 0);
  }, true);
});
