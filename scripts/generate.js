/******************************************************************************
 * Author: Frank Derry Wanye
 * Date: 11 Mar 2017
 * Loads example output from file and displays it in given text box.
 *****************************************************************************/

/**
 * Send ajax request to given URL, place result in given div
 */
var sendRequest = function(url, div) {
    div.innerHTML = "Waiting for a response from Terry...";

    var request = new XMLHttpRequest();

    request.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            var response = request.responseText;
            var outputs = response.split("\n");
            console.log("Received input of length " + outputs.length);
            var i = 0;
            while (i < outputs.length) {
                if (outputs[i].indexOf('fuck') >= 0 || outputs[i].length == 0) {
                    outputs.splice(i, 1);
                } else {
                    i++;
                }
            }
            console.log("After filtering, input length was " + outputs.length);
            var index = Math.floor(Math.random() * outputs.length);
            div.innerHTML = outputs[index];
        }
    };

    request.open("GET", url, true);
    request.send();

};


/**
 * Assign functionality to buttons
 */
document.addEventListener("readystatechange", function() {

    if (document.readyState === "interactive") {

        console.log("Document is interactive.");

        //=====================================================================

        console.log("Setting up variables.");

        var sentenceDiv = document.getElementById("sentences");
        var sentenceButton = sentenceDiv.getElementsByClassName("button")[0];
        var sentenceBox = sentenceDiv.getElementsByClassName("container")[0];

        var paragraphDiv = document.getElementById("paragraphs");
        var paragraphButton = paragraphDiv.getElementsByClassName("button")[0];
        var paragraphBox = paragraphDiv.getElementsByClassName("container")[0];

        var storyDiv = document.getElementById("stories");
        var storyButton = storyDiv.getElementsByClassName("button")[0];
        var storyBox = storyDiv.getElementsByClassName("container")[0];

        //=====================================================================

        console.log("Adding button functionality.")

        sentenceButton.onclick = function() {
            console.log("Sentence requested.");
            sendRequest("/results/f_sentences.txt", sentenceBox);
        };

        paragraphButton.onclick = function() {
            console.log("Paragraph requested.");
            sendRequest("/results/f_paragraphs.txt", paragraphBox);
        };

        storyButton.onclick = function() {
            console.log("Story requested.");
            sendRequest("/results/f_stories.txt", storyBox);
        };

    }

});
