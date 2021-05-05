const http = require('http');
var fs = require('fs');
const { spawn } = require('child_process');

const hostname = '127.0.0.1';
const port = 3000;

const server = http.createServer((req, res) => {
  fs.readFile('.' + req.url, function(err, data) {
    if (err) {
      res.writeHead(404);
      res.end(JSON.stringify(err));
      return;
    }
    res.writeHead(200)
    res.end(data);
  });
});

server.listen(port, hostname, () => {
  console.log(`running at http://${hostname}:${port}/`);
});

const python = spawn('python', ['jack-nn.py']);

var dataToSend = '';
python.stdout.on('data', function (data) {
  console.log('Pipe data from python script ...');
  dataToSend = data.toString();
});

python.on('close', (code) => {
  console.log(`child process close all stdio with code ${code}`);
  // send data to browser
  console.log(dataToSend);
});
