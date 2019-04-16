// modules are defined as an array
// [ module function, map of requires ]
//
// map of requires is short require name -> numeric require
//
// anything defined in a previous bundle is accessed via the
// orig method which is the require for previous bundles
parcelRequire = (function (modules, cache, entry, globalName) {
  // Save the require from previous bundle to this closure if any
  var previousRequire = typeof parcelRequire === 'function' && parcelRequire;
  var nodeRequire = typeof require === 'function' && require;

  function newRequire(name, jumped) {
    if (!cache[name]) {
      if (!modules[name]) {
        // if we cannot find the module within our internal map or
        // cache jump to the current global require ie. the last bundle
        // that was added to the page.
        var currentRequire = typeof parcelRequire === 'function' && parcelRequire;
        if (!jumped && currentRequire) {
          return currentRequire(name, true);
        }

        // If there are other bundles on this page the require from the
        // previous one is saved to 'previousRequire'. Repeat this as
        // many times as there are bundles until the module is found or
        // we exhaust the require chain.
        if (previousRequire) {
          return previousRequire(name, true);
        }

        // Try the node require function if it exists.
        if (nodeRequire && typeof name === 'string') {
          return nodeRequire(name);
        }

        var err = new Error('Cannot find module \'' + name + '\'');
        err.code = 'MODULE_NOT_FOUND';
        throw err;
      }

      localRequire.resolve = resolve;
      localRequire.cache = {};

      var module = cache[name] = new newRequire.Module(name);

      modules[name][0].call(module.exports, localRequire, module, module.exports, this);
    }

    return cache[name].exports;

    function localRequire(x){
      return newRequire(localRequire.resolve(x));
    }

    function resolve(x){
      return modules[name][1][x] || x;
    }
  }

  function Module(moduleName) {
    this.id = moduleName;
    this.bundle = newRequire;
    this.exports = {};
  }

  newRequire.isParcelRequire = true;
  newRequire.Module = Module;
  newRequire.modules = modules;
  newRequire.cache = cache;
  newRequire.parent = previousRequire;
  newRequire.register = function (id, exports) {
    modules[id] = [function (require, module) {
      module.exports = exports;
    }, {}];
  };

  var error;
  for (var i = 0; i < entry.length; i++) {
    try {
      newRequire(entry[i]);
    } catch (e) {
      // Save first error but execute all entries
      if (!error) {
        error = e;
      }
    }
  }

  if (entry.length) {
    // Expose entry point to Node, AMD or browser globals
    // Based on https://github.com/ForbesLindesay/umd/blob/master/template.js
    var mainExports = newRequire(entry[entry.length - 1]);

    // CommonJS
    if (typeof exports === "object" && typeof module !== "undefined") {
      module.exports = mainExports;

    // RequireJS
    } else if (typeof define === "function" && define.amd) {
     define(function () {
       return mainExports;
     });

    // <script>
    } else if (globalName) {
      this[globalName] = mainExports;
    }
  }

  // Override the current require with this new one
  parcelRequire = newRequire;

  if (error) {
    // throw error from earlier, _after updating parcelRequire_
    throw error;
  }

  return newRequire;
})({"ponggame.js":[function(require,module,exports) {
function asyncGeneratorStep(gen, resolve, reject, _next, _throw, key, arg) { try { var info = gen[key](arg); var value = info.value; } catch (error) { reject(error); return; } if (info.done) { resolve(value); } else { Promise.resolve(value).then(_next, _throw); } }

function _asyncToGenerator(fn) { return function () { var self = this, args = arguments; return new Promise(function (resolve, reject) { var gen = fn.apply(self, args); function _next(value) { asyncGeneratorStep(gen, resolve, reject, _next, _throw, "next", value); } function _throw(err) { asyncGeneratorStep(gen, resolve, reject, _next, _throw, "throw", err); } _next(undefined); }); }; }

function _toConsumableArray(arr) { return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _nonIterableSpread(); }

function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance"); }

function _iterableToArray(iter) { if (Symbol.iterator in Object(iter) || Object.prototype.toString.call(iter) === "[object Arguments]") return Array.from(iter); }

function _arrayWithoutHoles(arr) { if (Array.isArray(arr)) { for (var i = 0, arr2 = new Array(arr.length); i < arr.length; i++) { arr2[i] = arr[i]; } return arr2; } }

// initial model definition
var model = tf.sequential();
model.add(tf.layers.dense({
  units: 256,
  inputShape: [8]
})); //input is a 1x8

model.add(tf.layers.dense({
  units: 512,
  inputShape: [256],
  activation: "sigmoid"
}));
model.add(tf.layers.dense({
  units: 256,
  inputShape: [512],
  activation: "sigmoid"
}));
model.add(tf.layers.dense({
  units: 3,
  inputShape: [256]
})); //returns a 1x3

var learningRate = 0.001;
var optimizer = tf.train.adam(learningRate);
model.compile({
  loss: "meanSquaredError",
  optimizer: optimizer
}); //animation of the pong game code

var animate = window.requestAnimationFrame || window.webkitRequestAnimationFrame || window.mozRequestAnimationFrame || function (callback) {
  window.setTimeout(callback, 1000 / 60);
}; // variables for pong game.


var canvas = document.createElement("canvas");
var width = 400;
var height = 600;
canvas.width = width;
canvas.height = height;
var context = canvas.getContext("2d");
var player = new Player();
var computer = new Computer();
var ball = new Ball(200, 300);
var ai = new AI();
var keysDown = {}; //from pong code:

var render = function render() {
  context.fillStyle = "#000000";
  context.fillRect(0, 0, width, height);
  player.render();
  computer.render();
  ball.render();
}; //from pong code:


var update = function update() {
  player.update();

  if (computer.ai_plays) {
    move = ai.predict_move();
    computer.ai_update(move);
  } else computer.update(ball);

  ball.update(player.paddle, computer.paddle);
  ai.save_data(player.paddle, computer.paddle, ball);
}; //from pong code:


var step = function step() {
  update();
  render();
  animate(step);
}; //from pong code:


function Paddle(x, y, width, height) {
  this.x = x;
  this.y = y;
  this.width = width;
  this.height = height;
  this.x_speed = 0;
  this.y_speed = 0;
} //from pong code:


Paddle.prototype.render = function () {
  context.fillStyle = "#59a6ff";
  context.fillRect(this.x, this.y, this.width, this.height);
}; //from pong code:


Paddle.prototype.move = function (x, y) {
  this.x += x;
  this.y += y;
  this.x_speed = x;
  this.y_speed = y;

  if (this.x < 0) {
    this.x = 0;
    this.x_speed = 0;
  } else if (this.x + this.width > 400) {
    this.x = 400 - this.width;
    this.x_speed = 0;
  }
}; //from pong code:


function Computer() {
  this.paddle = new Paddle(175, 10, 50, 10);
  this.ai_plays = false;
} //from pong code:


Computer.prototype.render = function () {
  this.paddle.render();
}; //from pong code:


Computer.prototype.update = function (ball) {
  var x_pos = ball.x;
  var diff = -(this.paddle.x + this.paddle.width / 2 - x_pos);

  if (diff < 0 && diff < -4) {
    diff = -5;
  } else if (diff > 0 && diff > 4) {
    diff = 5;
  }

  this.paddle.move(diff, 0);

  if (this.paddle.x < 0) {
    this.paddle.x = 0;
  } else if (this.paddle.x + this.paddle.width > 400) {
    this.paddle.x = 400 - this.paddle.width;
  }
}; // Custom code. Depending on what
// move passed here, we move the computer 4x.
// Network output is either -1, 0, or 1 (left, stay, right)


Computer.prototype.ai_update = function () {
  var move = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 0;
  this.paddle.move(4 * move, 0);
}; //from pong code:


function Player() {
  this.paddle = new Paddle(175, 580, 50, 10);
} //from pong code:


Player.prototype.render = function () {
  this.paddle.render();
}; //from pong code:


Player.prototype.update = function () {
  for (var key in keysDown) {
    var value = Number(key);

    if (value == 37) {
      this.paddle.move(-4, 0);
    } else if (value == 39) {
      this.paddle.move(4, 0);
    } else {
      this.paddle.move(0, 0);
    }
  }
}; //from pong code:


function Ball(x, y) {
  this.x = x;
  this.y = y;
  this.x_speed = 0;
  this.y_speed = 3;
} //from pong code:


Ball.prototype.render = function () {
  context.beginPath();
  context.arc(this.x, this.y, 5, 2 * Math.PI, false);
  context.fillStyle = "#ddff59";
  context.fill();
}; //from pong code:


Ball.prototype.update = function (paddle1, paddle2, new_turn) {
  this.x += this.x_speed;
  this.y += this.y_speed;
  var top_x = this.x - 5;
  var top_y = this.y - 5;
  var bottom_x = this.x + 5;
  var bottom_y = this.y + 5;

  if (this.x - 5 < 0) {
    this.x = 5;
    this.x_speed = -this.x_speed;
  } else if (this.x + 5 > 400) {
    this.x = 395;
    this.x_speed = -this.x_speed;
  }

  if (this.y < 0 || this.y > 600) {
    this.x_speed = 0;
    this.y_speed = 3;
    this.x = 200;
    this.y = 300;
    ai.new_turn();
  }

  if (top_y > 300) {
    if (top_y < paddle1.y + paddle1.height && bottom_y > paddle1.y && top_x < paddle1.x + paddle1.width && bottom_x > paddle1.x) {
      this.y_speed = -3;
      this.x_speed += paddle1.x_speed / 2;
      this.y += this.y_speed;
    }
  } else {
    if (top_y < paddle2.y + paddle2.height && bottom_y > paddle2.y && top_x < paddle2.x + paddle2.width && bottom_x > paddle2.x) {
      this.y_speed = 3;
      this.x_speed += paddle2.x_speed / 2;
      this.y += this.y_speed;
    }
  }
}; // Custom code:
// stores data for ai.


function AI() {
  this.previous_data = null;
  this.training_data = [[], [], []];
  this.last_data_object = null;
  this.turn = 0;
  this.grab_data = true;
  this.flip_table = true;
} // Custom code:
// This code is responsible for saving data per frame


AI.prototype.save_data = function (player, computer, ball) {
  if (!this.grab_data) return; // If this is the very first frame (no prior data):

  if (this.previous_data == null) {
    data = this.flip_table ? [width - computer.x, width - player.x, width - ball.x, height - ball.y] : [player.x, computer.x, ball.x, ball.y];
    this.previous_data = data;
    return;
  } // table is rotated to learn from player, but apply to computer position:


  if (this.flip_table) {
    data_xs = [width - computer.x, width - player.x, width - ball.x, height - ball.y];
    index = width - player.x > this.previous_data[1] ? 0 : width - player.x == this.previous_data[1] ? 1 : 2;
  } else {
    data_xs = [player.x, computer.x, ball.x, ball.y];
    index = player.x < this.previous_data[0] ? 0 : player.x == this.previous_data[0] ? 1 : 2;
  }

  this.last_data_object = [].concat(_toConsumableArray(this.previous_data), _toConsumableArray(data_xs));
  this.training_data[index].push(this.last_data_object);
  this.previous_data = data_xs;
}; // Custom code:
// deciding whether to play as ai


AI.prototype.new_turn = function () {
  this.previous_data = null;
  this.turn++;
  console.log("new turn: " + this.turn); //hm games til train?

  if (this.turn > 5) {
    debugger;
    this.train();
    computer.ai_plays = true;
    this.reset();
  }
}; // Custom code:
// empty training data to start clean


AI.prototype.reset = function () {
  this.previous_data = null;
  this.training_data = [[], [], []];
  this.turn = 0;
}; // Custom code:
// trains a model


AI.prototype.train = function () {
  console.log("balancing"); //shuffle attempt

  len = Math.min(this.training_data[0].length, this.training_data[1].length, this.training_data[2].length);

  if (!len) {
    console.log("nothing to train");
    return;
  }

  data_xs = [];
  data_ys = [];

  for (i = 0; i < 3; i++) {
    var _data_xs, _data_ys;

    (_data_xs = data_xs).push.apply(_data_xs, _toConsumableArray(this.training_data[i].slice(0, len)));

    (_data_ys = data_ys).push.apply(_data_ys, _toConsumableArray(Array(len).fill([i == 0 ? 1 : 0, i == 1 ? 1 : 0, i == 2 ? 1 : 0])));
  }

  console.log("training");
  var xs = tf.tensor(data_xs);
  var ys = tf.tensor(data_ys);

  _asyncToGenerator(
  /*#__PURE__*/
  regeneratorRuntime.mark(function _callee() {
    var result;
    return regeneratorRuntime.wrap(function _callee$(_context) {
      while (1) {
        switch (_context.prev = _context.next) {
          case 0:
            console.log("training2");
            _context.next = 3;
            return model.fit(xs, ys);

          case 3:
            result = _context.sent;
            console.log(result);

          case 5:
          case "end":
            return _context.stop();
        }
      }
    }, _callee);
  }))();

  console.log("trained");
}; // Custom code:


AI.prototype.predict_move = function () {
  console.log("predicting");

  if (this.last_data_object != null) {
    //use this.last_data_object for input data
    //do prediction here
    //return -1/0/1
    prediction = model.predict(tf.tensor([this.last_data_object]));
    return tf.argMax(prediction, 1).dataSync() - 1;
  }
}; // Original pong code:


document.body.appendChild(canvas);
animate(step);
window.addEventListener("keydown", function (event) {
  keysDown[event.keyCode] = true;
});
window.addEventListener("keyup", function (event) {
  delete keysDown[event.keyCode];
});
},{}],"../../../../../AppData/Roaming/npm/node_modules/parcel-bundler/src/builtins/hmr-runtime.js":[function(require,module,exports) {
var global = arguments[3];
var OVERLAY_ID = '__parcel__error__overlay__';
var OldModule = module.bundle.Module;

function Module(moduleName) {
  OldModule.call(this, moduleName);
  this.hot = {
    data: module.bundle.hotData,
    _acceptCallbacks: [],
    _disposeCallbacks: [],
    accept: function (fn) {
      this._acceptCallbacks.push(fn || function () {});
    },
    dispose: function (fn) {
      this._disposeCallbacks.push(fn);
    }
  };
  module.bundle.hotData = null;
}

module.bundle.Module = Module;
var checkedAssets, assetsToAccept;
var parent = module.bundle.parent;

if ((!parent || !parent.isParcelRequire) && typeof WebSocket !== 'undefined') {
  var hostname = "" || location.hostname;
  var protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  var ws = new WebSocket(protocol + '://' + hostname + ':' + "51114" + '/');

  ws.onmessage = function (event) {
    checkedAssets = {};
    assetsToAccept = [];
    var data = JSON.parse(event.data);

    if (data.type === 'update') {
      var handled = false;
      data.assets.forEach(function (asset) {
        if (!asset.isNew) {
          var didAccept = hmrAcceptCheck(global.parcelRequire, asset.id);

          if (didAccept) {
            handled = true;
          }
        }
      }); // Enable HMR for CSS by default.

      handled = handled || data.assets.every(function (asset) {
        return asset.type === 'css' && asset.generated.js;
      });

      if (handled) {
        console.clear();
        data.assets.forEach(function (asset) {
          hmrApply(global.parcelRequire, asset);
        });
        assetsToAccept.forEach(function (v) {
          hmrAcceptRun(v[0], v[1]);
        });
      } else {
        window.location.reload();
      }
    }

    if (data.type === 'reload') {
      ws.close();

      ws.onclose = function () {
        location.reload();
      };
    }

    if (data.type === 'error-resolved') {
      console.log('[parcel] ✨ Error resolved');
      removeErrorOverlay();
    }

    if (data.type === 'error') {
      console.error('[parcel] 🚨  ' + data.error.message + '\n' + data.error.stack);
      removeErrorOverlay();
      var overlay = createErrorOverlay(data);
      document.body.appendChild(overlay);
    }
  };
}

function removeErrorOverlay() {
  var overlay = document.getElementById(OVERLAY_ID);

  if (overlay) {
    overlay.remove();
  }
}

function createErrorOverlay(data) {
  var overlay = document.createElement('div');
  overlay.id = OVERLAY_ID; // html encode message and stack trace

  var message = document.createElement('div');
  var stackTrace = document.createElement('pre');
  message.innerText = data.error.message;
  stackTrace.innerText = data.error.stack;
  overlay.innerHTML = '<div style="background: black; font-size: 16px; color: white; position: fixed; height: 100%; width: 100%; top: 0px; left: 0px; padding: 30px; opacity: 0.85; font-family: Menlo, Consolas, monospace; z-index: 9999;">' + '<span style="background: red; padding: 2px 4px; border-radius: 2px;">ERROR</span>' + '<span style="top: 2px; margin-left: 5px; position: relative;">🚨</span>' + '<div style="font-size: 18px; font-weight: bold; margin-top: 20px;">' + message.innerHTML + '</div>' + '<pre>' + stackTrace.innerHTML + '</pre>' + '</div>';
  return overlay;
}

function getParents(bundle, id) {
  var modules = bundle.modules;

  if (!modules) {
    return [];
  }

  var parents = [];
  var k, d, dep;

  for (k in modules) {
    for (d in modules[k][1]) {
      dep = modules[k][1][d];

      if (dep === id || Array.isArray(dep) && dep[dep.length - 1] === id) {
        parents.push(k);
      }
    }
  }

  if (bundle.parent) {
    parents = parents.concat(getParents(bundle.parent, id));
  }

  return parents;
}

function hmrApply(bundle, asset) {
  var modules = bundle.modules;

  if (!modules) {
    return;
  }

  if (modules[asset.id] || !bundle.parent) {
    var fn = new Function('require', 'module', 'exports', asset.generated.js);
    asset.isNew = !modules[asset.id];
    modules[asset.id] = [fn, asset.deps];
  } else if (bundle.parent) {
    hmrApply(bundle.parent, asset);
  }
}

function hmrAcceptCheck(bundle, id) {
  var modules = bundle.modules;

  if (!modules) {
    return;
  }

  if (!modules[id] && bundle.parent) {
    return hmrAcceptCheck(bundle.parent, id);
  }

  if (checkedAssets[id]) {
    return;
  }

  checkedAssets[id] = true;
  var cached = bundle.cache[id];
  assetsToAccept.push([bundle, id]);

  if (cached && cached.hot && cached.hot._acceptCallbacks.length) {
    return true;
  }

  return getParents(global.parcelRequire, id).some(function (id) {
    return hmrAcceptCheck(global.parcelRequire, id);
  });
}

function hmrAcceptRun(bundle, id) {
  var cached = bundle.cache[id];
  bundle.hotData = {};

  if (cached) {
    cached.hot.data = bundle.hotData;
  }

  if (cached && cached.hot && cached.hot._disposeCallbacks.length) {
    cached.hot._disposeCallbacks.forEach(function (cb) {
      cb(bundle.hotData);
    });
  }

  delete bundle.cache[id];
  bundle(id);
  cached = bundle.cache[id];

  if (cached && cached.hot && cached.hot._acceptCallbacks.length) {
    cached.hot._acceptCallbacks.forEach(function (cb) {
      cb();
    });

    return true;
  }
}
},{}]},{},["../../../../../AppData/Roaming/npm/node_modules/parcel-bundler/src/builtins/hmr-runtime.js","ponggame.js"], null)
//# sourceMappingURL=/ponggame.e2f5a9bb.js.map