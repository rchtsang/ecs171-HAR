async function getFile(event) {
  var file = event.target.files[0];
  document.getElementById('output').textContent = 'working...'
  file.text().then(async function(result) {
    const http = new XMLHttpRequest();
    const url='/';
    http.open("POST", url);
    http.send(result);
    http.onreadystatechange = (e) => {
      console.log(http.responseText);
      document.getElementById('output').textContent = http.responseText;
    }
  });
}