const express = require('express');

const router = express.Router();

router.post('/login', function(req, res) {
  console.log(req.body.username);
  console.log(req.body.password); 
  if (req.body.username === 'admin' && req.body.password === 'admin') {
    require('crypto').randomBytes(48, function(err, buffer) {
      require('../db').addminToken = buffer.toString('hex');
      res.send({status: 'success', role: 'admin', token: require('../db').addminToken});
    });
  }
  else if (req.body.username === 'employee' && req.body.password === 'employee') {
    require('crypto').randomBytes(48, function(err, buffer) {
      require('../db').employeeToken = buffer.toString('hex');
      res.send({status: 'success', role: 'employee', token: require('../db').employeeToken});
    });
  }
  else {
    res.send({status: 'error', message: 'access denied.'});
  }
});

module.exports = router;
