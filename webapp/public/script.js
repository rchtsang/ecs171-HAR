async function getFile(event) {
  var file = event.target.files[0];
  document.getElementById('output').textContent = 'working...'
  file.text().then(async function(result) {
    console.log(result);
  });
}