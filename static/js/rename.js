function applyRename(){
    var transcriptElem = document.getElementById("transcript");
    var transcriptText = transcriptElem.value;
    var inputs = document.querySelectorAll('input[type="text"]');
    inputs.forEach(function(input) {
        var oldName = input.name;
        var newName = input.value;
        if(newName){
            var regex = new RegExp(oldName, "g");
            transcriptText = transcriptText.replace(regex, newName);
        }
    });
    transcriptElem.value = transcriptText;
}
