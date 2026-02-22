(function () {
  const form = document.getElementById('diagnose-form');
  const input = document.getElementById('symptoms-input');
  const submitBtn = document.getElementById('submit-btn');
  const messagesEl = document.getElementById('messages');
  const welcomeEl = document.getElementById('welcome');

  function hideWelcome() {
    messagesEl.classList.add('has-messages');
  }

  function appendMessage(role, contentEl) {
    hideWelcome();
    const wrap = document.createElement('div');
    wrap.className = 'message message-' + role;
    const label = role === 'user' ? 'You' : 'Assistant';
    wrap.innerHTML = '<div class="message-label">' + label + '</div>';
    wrap.querySelector('.message-label').after(contentEl);
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return wrap;
  }

  function createUserBubble(text) {
    const div = document.createElement('div');
    div.className = 'message-content';
    div.textContent = text;
    return div;
  }

  function createDiagnosisCards(diagnoses) {
    const block = document.createElement('div');
    block.className = 'diagnoses-block';
    let html = '<div class="diagnoses-title">Probable diagnoses (ICD-10)</div><div class="diagnosis-list">';
    for (const d of diagnoses) {
      const name = escapeHtml(d.diagnosis || '—');
      const code = escapeHtml(d.icd10_code || '');
      const expl = escapeHtml(d.explanation || '');
      html +=
        '<div class="diagnosis-card rank-' + d.rank + '">' +
        '<div class="diagnosis-header">' +
        '<span class="diagnosis-rank">' + d.rank + '</span>' +
        '<span class="diagnosis-name">' + name + '</span>' +
        (code ? '<span class="icd-badge">' + code + '</span>' : '') +
        '</div>' +
        (expl ? '<p class="diagnosis-explanation">' + expl + '</p>' : '') +
        '</div>';
    }
    html += '</div>';
    block.innerHTML = html;
    const content = document.createElement('div');
    content.className = 'message-content';
    content.appendChild(block);
    return content;
  }

  function createLoadingBubble() {
    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML =
      '<div class="typing-dots">' +
      '<span></span><span></span><span></span>' +
      '</div>';
    return content;
  }

  function createErrorBubble(message) {
    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = '<p class="error-text">' + escapeHtml(message) + '</p>';
    return content;
  }

  function escapeHtml(s) {
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
  }

  async function diagnose(symptoms) {
    const text = (symptoms || '').trim();
    if (!text) return;

    appendMessage('user', createUserBubble(text));
    input.value = '';

    const loadingWrap = appendMessage('assistant', createLoadingBubble());
    submitBtn.disabled = true;

    try {
      const res = await fetch('/diagnose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symptoms: text }),
      });

      loadingWrap.remove();

      if (!res.ok) {
        const errBody = await res.text();
        let errMsg = 'Request failed';
        try {
          const j = JSON.parse(errBody);
          if (j.detail) errMsg = Array.isArray(j.detail) ? j.detail.map(d => d.msg || d).join(' ') : String(j.detail);
        } catch (_) {
          if (errBody) errMsg = errBody.slice(0, 200);
        }
        appendMessage('assistant', createErrorBubble(errMsg));
        return;
      }

      const data = await res.json();
      const diagnoses = data.diagnoses || [];
      appendMessage('assistant', createDiagnosisCards(diagnoses));
    } catch (e) {
      loadingWrap.remove();
      appendMessage('assistant', createErrorBubble(e.message || 'Network error'));
    } finally {
      submitBtn.disabled = false;
    }
  }

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    diagnose(input.value);
  });

  document.querySelectorAll('.suggestion-pill').forEach(function (btn) {
    btn.addEventListener('click', function () {
      const s = this.getAttribute('data-symptoms') || '';
      input.value = s;
      input.focus();
      diagnose(s);
    });
  });

  input.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
})();
