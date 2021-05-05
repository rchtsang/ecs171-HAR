const { spawn } = require('child_process');

const express = require('express');
const app = express();
const port = 3000;

app.use(express.static('public'));

app.listen(port, () => {
  console.log(`Example app listening at http://localhost:${port}`);
});

const python = spawn('python', ['models/nn.py']);

var dataToSend = 'no data';
python.stdout.on('data', function (data) {
  console.log('Pipe data from python script ...');
  dataToSend = data.toString();
});

python.on('close', (code) => {
  console.log(`child process close all stdio with code ${code}`);
  console.log(dataToSend);
});
