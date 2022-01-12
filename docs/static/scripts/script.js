let content = document.getElementById("content");
let digits = [];
console.log(content)

for(let i = 0; i < 10; i++){
    let digit = {
        container: document.createElement("div"),
        content : {
            bar: document.createElement("div"),
            label: document.createElement("div"),
        }
    }

    digit.container.id = i;
    digit.content.bar.id = "bar_" + i;
    digit.content.label.id = "label_" + i;

    digit.content.bar.classList.add("bar");
    digit.content.label.classList.add("label");

    content.appendChild(digit.container);
    digit.container.appendChild(digit.content.bar);
    digit.container.appendChild(digit.content.label);

    digit.content.label.innerHTML = i;

    digits.push(digit);
}

let drawing = false;

let canvas = document.getElementById("canvas");
let cur_pos;
let prev_pos;

let model = tf.loadGraphModel('static/lememe/model.json');


canvas.onmousemove = (e) => {
    if(prev_pos == undefined){
        prev_pos = [e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop];
    }
    else{
        prev_pos = cur_pos;
    }
    cur_pos = [e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop];

    if(drawing){
        let ctx = canvas.getContext('2d');
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(prev_pos[0], prev_pos[1]);
        ctx.lineTo(cur_pos[0], cur_pos[1]);
        ctx.strokeStyle = "white";
        ctx.lineWidth = 20;
        ctx.stroke();
        ctx.closePath();
    }
}

canvas.onmousedown = (e) => { drawing = true; clear();}
canvas.onmouseup = (e) => {drawing = false; eval(); };

function clear(){
    let ctx = canvas.getContext('2d');
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function eval(){
    let ctx = canvas.getContext('2d');
    let image = ctx.getImageData(0, 0, canvas.width, canvas.height);
    image = tf.browser.fromPixels(image).resizeBilinear([28,28]).mean(2).toFloat().expandDims(0).expandDims(0);
    let num_pred;
    model.then(function (res) {
        num_pred = res.predict(image).argMax(-1).dataSync()[0];

        for(let i = 0; i < 10; i++){
            let bar = document.getElementById("bar_"+ i);
            let label = document.getElementById("label_"+ i);
            // console.log(bar);
            if(i == num_pred){
                bar.classList.add("argmax");
                label.classList.add("argmax");
            }else{
                bar.classList.remove("argmax");
                label.classList.remove("argmax");
            }
        }
        console.log(num_pred);
    }, function (err) {
        console.log(err);
    });
}
