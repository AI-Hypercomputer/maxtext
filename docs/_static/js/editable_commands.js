/**
 * Handles inline editable commands in documentation.
 * Replaces placeholders in code blocks with inline input fields.
 */
document.addEventListener('DOMContentLoaded', () => {
  const codeBlocks = document.querySelectorAll('div.highlight-sh pre, div.highlight-bash pre, div.highlight-default pre');

  codeBlocks.forEach(block => {

    const originalHTML = block.innerHTML;

    const placeholders = [
      "<batch size per device>",
      "<bucket>",
      "<cluster name>",
      "<data columns to train on>",
      "<Data Columns to Train on>",
      "<data split for train>",
      "<Data Split for Train>",
      "<dataset path>",
      "<Docker Image Name>",
      "<Fine-Tuning Steps>",
      "<Flag to lazy load>",
      "<Flag to use ocdbt>",
      "<Flag to use zarr3>",
      "<folder>",
      "<gcs path for MaxText checkpoint>",
      "<GCS Path for Output/Logs>",
      "<GCS for dataset>",
      "<GCP project ID>",
      "<GCP zone>",
      "<gke version>",
      "<GKE Cluster Zone>",
      "<Google Cloud Project ID>",
      "<Hugging Face Access Token>",
      "<Hugging Face access token>",
      "<Hugging Face Dataset Name>",
      "<Hugging Face dataset name>",
      "<Hugging Face Model>",
      "<Hugging Face Model to be converted to MaxText>",
      "<MaxText Model>",
      "<MaxText model name>",
      "<Model Name>",
      "<model name>",
      "<Model Tokenizer>",
      "<name for this run>",
      "<Name for this run>",
      "<Name of GKE Cluster>",
      "<Name of Workload>",
      "<number of fine-tuning steps to run>",
      "<number of slices>",
      "<output directory to store Hugging Face checkpoint>",
      "<output directory to store MaxText checkpoint>",
      "<output directory to store run logs>",
      "<path to Hugging Face checkpoint>",
      "<path/to/gcr.io>",
      "<project id>",
      "<project ID>",
      "<project>",
      "<ramdisk size>",
      "<steps>",
      "<the number of chips per VM>",
      "<Tokenizer>",
      "<tokenizer path>",
      "<TPU Type>",
      "<virtual env name>",
      "<your virtual env name>",
      "<your zone>",
      "<YOUR WORKLOAD NAME>",
      "<zone>",
      "<zone name>"
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
