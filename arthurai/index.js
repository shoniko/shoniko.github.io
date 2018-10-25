const MODEL_URL = 'model/tensorflowjs_model.pb';
const WEIGHTS_URL = 'model/weights_manifest.json';

let model = null;

const charIndex = {
    'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'g': 8, 'h': 9, 'i': 10,
    'j': 11, 'k': 12, 'l': 13, 'm': 14, 'n': 15, 'o': 16, 'p': 17, 'q': 18,
    'r': 19, 's': 20, 't': 21, 'u': 22, 'v': 23, 'w': 24, 'x': 25, 'y': 26,
    'z': 27, '1': 28, '2': 29, '3': 30, '4': 31, '5': 32, '6': 33, '7': 34,
    '8': 35, '9': 36, '0': 37, ':': 38, ';': 39, '/': 40, '?': 41, '!': 42,
    '=': 43, '+': 44, '.': 45, ',': 46, '(': 47, ')': 48, '[': 49, ']': 50,
    '-': 51, '`': 52, '*': 53, '_': 54, '\\': 55, '|': 56, '~': 57
}

let typeIndex = {};

function onTfLoaded() {
    typeIndex =  {
        'SCRIPT':           tf.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        'SUBDOCUMENT':      tf.tensor([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        'IMAGE':            tf.tensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        'XMLHTTPREQUEST':   tf.tensor([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]),
        'FONT':             tf.tensor([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]),
        'DOCUMENT':         tf.tensor([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]),
        'STYLESHEET':       tf.tensor([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),
        'OTHER':            tf.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]),
        'PING':             tf.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]),
        'WEBSOCKET':        tf.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]),
        'MEDIA':            tf.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]),
        'OBJECT':           tf.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    }

    loadModel();
}

function stringToTensor(text, padLength) {
    // Based on https://github.com/tensorflow/tfjs-examples/blob/master/sentiment/index.js#L60
    const inputText = text.trim().toLowerCase();
    const inputBuffer = tf.buffer([1, padLength], 'float32');
    for (let i = 0; i < inputText.length; ++i)
    {
      const chr = inputText[i];
      inputBuffer.set(charIndex[chr], 0, i);
    }
    return inputBuffer.toTensor();
}

function typeToTensor(inputType) {
    return typeIndex[inputType];
}

function loadScript(url, callback) {
    var script = document.createElement("script")
    script.type = "text/javascript";

    script.onload = function() {
        if (callback) {
          callback();
        }
    };

    script.src = url;
    document.getElementsByTagName("head")[0].appendChild(script);
}

async function loadModel() {
    model = await tf.loadFrozenModel(MODEL_URL, WEIGHTS_URL);
}

function predict() {
    const url = document.getElementById("url").value;
    const domain = document.getElementById("domain").value;
    const type = document.getElementById("type").value;
    const urlTensor = stringToTensor(url, 512);
    const domainTensor = stringToTensor(domain, 100);
    const typeTensor = typeToTensor(type);

    var t0 = performance.now();
    const result = model.predict({
        Placeholder: urlTensor,
        Placeholder_1: typeTensor,
        Placeholder_2: domainTensor
    }).dataSync();
    console.log(result);
    var t1 = performance.now();
    document.getElementById("result").textContent = "Model thinks there's " + result[0].toFixed(4) * 100 + "% chance it's an ad";
    document.getElementById("latency").textContent = "Latency: " + (t1 - t0) + "ms";
    blockitElement = document.getElementById("result-blockit");
    if (result[0] > 0.5) {
        blockitElement.textContent = "BLOCK IT!";
        blockitElement.className = "blockit";
    } else {
        blockitElement.textContent = "Don't block it";
        blockitElement.className = "dontblockit";
    }
}
function runTests() {
    const numOfIterations = 1000;
    const urlTensor = stringToTensor("https://google.com/ads.js", 512);
    const domainTensor = stringToTensor("adblockplus.org", 100);
    const typeTensor = typeToTensor("SCRIPT");
    var t0 = performance.now();
    model.predict({
        Placeholder: urlTensor,
        Placeholder_1: typeTensor,
        Placeholder_2: domainTensor
    }).print();
    var t1 = performance.now();
    console.log("Call to predict " + numOfIterations + " URLs took " + (t1 - t0) + " milliseconds.");
}

loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.13.0', onTfLoaded)
