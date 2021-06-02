const { spawn } = require('child_process');
const fs = require('fs')

const express = require('express');
const app = express();
const port = 3000;

app.use(express.static('public'));
app.use(express.text())

app.post('/', function (req, res) {
  fs.writeFile('./models/data.arff', req.body, { flag : 'w+' }, err => {
    if (err) {
      console.error(err)
      return
    }
    console.log('file written successfully');
    predict(res);
  })
});

app.listen(port, () => {
  console.log(`Example app listening at http://localhost:${port}`);
});

function predict(res) {
  const python = spawn('python', ['models/predict.py']);

  var dataToSend = 'no data';
  var err = 'no err';
  python.stdout.on('data', function (data) {
    dataToSend = data.toString();
  });

  python.stderr.on('data', function (data) {
    err = data.toString();
  });

  python.on('close', (code) => {
    console.log(`child process close all stdio with code ${code}`);
    console.log(dataToSend);
    console.log(err);
    res.send(dataToSend);
  });
}
