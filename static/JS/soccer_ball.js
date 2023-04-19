const ball = document.getElementById("ball");
const container = document.getElementById("container");
const resetButton = document.getElementById("reset-button");

let x = container.offsetWidth / 2 - ball.offsetWidth / 2;
let y = container.offsetHeight / 2 - ball.offsetHeight / 2;
let vx = 0;
let vy = 0;
let isMoving = false;

resetButton.addEventListener("click", reset);

function reset() {
  x = container.offsetWidth / 2 - ball.offsetWidth / 2;
  y = container.offsetHeight / 2 - ball.offsetHeight / 2;
  vx = 0;
  vy = 0;
  isMoving = false;
  ball.style.transform = "none";
  ball.addEventListener("mousedown", onMouseDown);
}

function onMouseDown(event) {
  const mouseX = event.clientX - x - ball.offsetWidth / 2;
  const mouseY = event.clientY - y - ball.offsetHeight / 2;
  const maxSpeed = 10;
  vx = (mouseX / ball.offsetWidth) * maxSpeed;
  vy = (mouseY / ball.offsetHeight) * maxSpeed;
  isMoving = true;
  ball.removeEventListener("mousedown", onMouseDown);
  requestAnimationFrame(update);
}

function update() {
  if (isMoving) {
    x += vx;
    y += vy;
    const ballBounds = ball.getBoundingClientRect();
    const containerBounds = container.getBoundingClientRect();
    if (ballBounds.left < containerBounds.left || ballBounds.right > containerBounds.right) {
      vx = -vx;
    }
    if (ballBounds.top < containerBounds.top || ballBounds.bottom > containerBounds.bottom) {
      vy = -vy;
    }
    ball.style.left = `${x}px`;
    ball.style.top = `${y}px`;
    ball.style.transform = `rotate(${(x + y) / 10}deg)`;
    requestAnimationFrame(update);
  }
}
