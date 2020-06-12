const express = require('express');
const XLSX = require('xlsx');
const fs = require('fs-extra');
const busboy = require('connect-busboy');
const spawn = require("child_process").spawn;

const router = express.Router();

router.use(busboy());

router.post('/forecast-load', function(req, res) {
  try {
    var fstream;
    req.pipe(req.busboy);
    console.log("file received 1");
    req.busboy.on('file', function (fieldname, file, filename) {
    try {
      console.log("file received 2");
      fstream = fs.createWriteStream('datasetW.xlsx');
      file.pipe(fstream);
      fstream.on('close', async function () {
        try {
          let child = spawn('venv/bin/python3', ['merge.py'], {cwd: '/home/deep-power-backend/'});
          child.stdout.setEncoding('utf-8');
          toSheetResult = new Array(2);
          toSheetResult[0] = new Array(24);
          toSheetResult[1] = new Array(24);
          let i = 0;
          for await (let data of child.stdout) {
            console.log(data);
            var tokens = data.split(/[\r\n]+/);
            for await (let token of tokens) {
              toSheetResult[0][i] = (i < 10 ? ("0" + i) : i) + ":00";
              toSheetResult[1][i] = token;
              i++;
            }
          }
          console.log("file processed,");
          var wb = XLSX.utils.book_new();
          wb.SheetNames.push("output");
          var ws = XLSX.utils.aoa_to_sheet(toSheetResult);
          wb.Sheets["output"] = ws;
          XLSX.writeFile(wb, 'results.xlsx');
          res.download('results.xlsx');
        }
        catch(ex) {
          console.log(ex);
        }
      });
    } catch(ex) {
      console.log(ex);
    }
  });
  } catch(ex) {
    console.log(ex);
  }
});

module.exports = router;
