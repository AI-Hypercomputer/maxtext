/**
 * Handles inline editable commands in documentation.
 * Replaces placeholders in code blocks with inline editable spans.
 * Using contenteditable spans avoids the "newline on copy" issue caused by inputs.
 */
document.addEventListener('DOMContentLoaded', () => {
  const codeBlocks = document.querySelectorAll('div.highlight-sh pre, div.highlight-bash pre, div.highlight-default pre');

  const placeholders = [
    "<BATCH_SIZE_PER_DEVICE>",
    "<CHIPS_PER_VM>",
    "<CKPT_PATH>",
    "<CLUSTER_NAME>",
    "<DATA_COLUMNS>",
    "<DATASET_NAME>",
    "<DATASET_PATH>",
    "<GCS_BUCKET>",
    "<HF_CKPT_PATH>",
    "<HF_MODEL>",
    "<HF_TOKEN>",
    "<IMAGE_NAME>",
    "<LAZY_LOAD>",
    "<MODEL_NAME>",
    "<NUM_SLICES>",
    "<POD_NAME>",
    "<PROJECT_ID>",
    "<RUN_NAME>",
    "<STEPS>",
    "<TPU_TYPE>",
    "<TRAIN_SPLIT>",
    "<VENV_NAME>",
    "<ZONE>"
  ];

  codeBlocks.forEach(block => {
    const originalHTML = block.innerHTML;
    let newHTML = originalHTML;

    placeholders.forEach(placeholder => {
      // 1. Create robust regex for this placeholder
      // Escape chars
      const escapeRegex = (string) => string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

      const htmlEscapedKey = placeholder
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

      let pattern = '';
      for (let i = 0; i < htmlEscapedKey.length; i++) {
        const char = htmlEscapedKey[i];
        // FIX: Avoid matching across our inserted spans by ignoring tags with contenteditable
        pattern += escapeRegex(char) + '(?:<(?!span[^>]*contenteditable)[^>]+>)*';
      }

      const regex = new RegExp(pattern, 'g');

      // Replace with a contenteditable span
      // The styling mimics an input field but remains strictly inline
      const spanHTML = `<span class="inline-input" contenteditable="true" spellcheck="false" data-placeholder="${placeholder}" style="border-bottom: 1px dashed #888; background: rgba(128, 128, 128, 0.15); padding: 0 4px; border-radius: 2px; outline: none;">${htmlEscapedKey}</span>`;

      newHTML = newHTML.replace(regex, spanHTML);
    });

    if (newHTML !== originalHTML) {
      block.innerHTML = newHTML;
    }
  });

  // Bind behavioral events to the newly created editable spans
  document.querySelectorAll('.inline-input').forEach(span => {

    // Auto-select the text when clicked, so the user can immediately type over the placeholder
    span.addEventListener('focus', function () {
      if (this.textContent === this.getAttribute('data-placeholder')) {
        const range = document.createRange();
        range.selectNodeContents(this);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
      }
    });

    // If the user deletes everything and clicks away, restore the original placeholder
    span.addEventListener('blur', function () {
      if (this.textContent.trim() === '') {
        this.textContent = this.getAttribute('data-placeholder');
      }
    });

    // Prevent 'Enter' from creating a messy multiline command block
    span.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        this.blur(); // Drop focus instead of breaking to a new line
      }
    });
  });
});
