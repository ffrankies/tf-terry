# Project Progress Journal

<!-- TOC -->

- [Getting Started](#getting-started)
    - [Major Happenings](#major-happenings)
    - [Roadblocks](#roadblocks)
    - [Questions](#questions)
- [Week 1](#week-1)
    - [Major Happenings](#major-happenings-1)
    - [Roadblocks](#roadblocks-1)
    - [Questions](#questions-1)

<!-- /TOC -->

## Getting Started

### Major Happenings

- Installed [Visual Studio Code](https://code.visualstudio.com/download) as my main code editor after having issues with python in Atom.
- Installed [tensorflow](https://www.tensorflow.org/install/) on my home laptop - this should be easier to use than Theano.
- Installed [bash on ubuntu on windows](https://msdn.microsoft.com/en-us/commandline/wsl/install_guide) because I didn't want to deal with Powershell.
- Broke my previous code for the Terry project into separate modules, to make reading and maintaining it easier.
- Set up [Codebeat](https://codebeat.co/) to analyze my code and suggest ways to improve it.
- After a lot of searching around, finally settled on a [tutorial](https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767) for creating RNNs in tensorflow. 
- Followed the tutorial through Part 3. At this point, I have:
    - An RNN that succesfully trains on pre-formatted text datasets (i.e., the training loss goes down with time).
    - The RNN uses the built-in API such that I have very little math code in my work. 
    - The LSTM cells from the tutorial are replaced with GRU cells, for they are simpler, require less computations, and apparently produce very similar results.

### Roadblocks

- I failed to get the GPU version of tensorflow to run on my laptop.
- Documentation for tensorflow can be confusing, and is not as extensive as the documentation for Theano.
- Unlike with Theano, I could not find a tensorflow tutorial that showed me how to do exactly what I needed it to (although, there are plenty of general RNN tutorials in tensorflow).
- Apparently, my tensorflow version wasn't compiled with certain options that could speed up training on my laptop.

### Questions

- How do I 'publish' my python modules such that I will be able to re-use them in the main project? At the moment, I'm thinking either publishing them via 'pip', or creating a git submodule out of them.

## Week 1

### Major Happenings

- Worked on getting my RNN to generate text.
- Gained a better understanding of how the RNN API in tensorflow works.
- Finally fully understood how the 'randomness' in numpy worked with the text generation.

### Roadblocks

- The RNN is generating duplicate gibberish, with text samples containing phrases like "I I I I I I I I I were here were were..."
- As usual, debugging a running RNN is difficult, although so far the tensorflow errors have been much easier to read than the Theano ones.
- When attempting to have the RNN learn where to put spaces, the RNN never once output a space token, despite it being the most common token in both the training and output data.
- It seems that the RNN API does not convert data to one-hot vectors automatically, like I thought it would. I may have to do that step manually. The good news is that this step may help with my text generation problem.

### Questions

- Is learning to use the tensorboard feature going to take a long time? And would it help me diagnose problems earlier?
- For the main Kindschi Fellowship project, how do we judge the success of the network? We can't really test it in a real-life scenario, so we might not actually know how useful this is.
- How do we deal with the network generalizing movement patterns? We will either have to manually give it 'seeded' movements to represent the first couple steps, or find a way to group the training data and have an extra 'feature' representing the data group. Grouping the data, however, simply based on location, is a task that would seem to require a separate neural network, and we may not have the time to design one.