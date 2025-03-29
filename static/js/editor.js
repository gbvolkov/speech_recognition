// Initialize Quill editor with provided transcript content
var quill = new Quill('#editor', {
    theme: 'snow'
});
quill.root.innerHTML = transcriptContent;

// Sidebar toggle functionality
document.getElementById("toggleRenameSidebar").onclick = function() {
    document.getElementById("renameSidebar").classList.toggle("active");
};

// Apply renaming functionality (from rename.js)
document.getElementById("applyRenameBtn").onclick = function() {
    applyRename();
    document.getElementById("renameSidebar").classList.remove("active");
};

// Helper function to submit content for download
function downloadContent(format) {
    const form = document.getElementById('editorForm');
    document.getElementById('html_content').value = quill.root.innerHTML;
    document.getElementById('format').value = format;
    form.submit();
}

// Event listeners for download buttons
document.getElementById('download-markup').onclick = function() {
    downloadContent('markup');
};

document.getElementById('download-word').onclick = function() {
    downloadContent('word');
};
