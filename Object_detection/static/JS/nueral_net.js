const svg = d3.select("#canvas");

const inputLayer = [
  { x: 50, y: 50 },
  { x: 50, y: 100 },
  { x: 50, y: 150 },
];

const hiddenLayer = [
  { x: 200, y: 50 },
  { x: 200, y: 100 },
  { x: 200, y: 150 },
];

const outputLayer = [
  { x: 350, y: 75 },
  { x: 350, y: 125 },
];

const links = [
  { source: inputLayer[0], target: hiddenLayer[0] },
  { source: inputLayer[0], target: hiddenLayer[1] },
  { source: inputLayer[0], target: hiddenLayer[2] },
  { source: inputLayer[1], target: hiddenLayer[0] },
  { source: inputLayer[1], target: hiddenLayer[1] },
  { source: inputLayer[1], target: hiddenLayer[2] },
  { source: inputLayer[2], target: hiddenLayer[0] },
  { source: inputLayer[2], target: hiddenLayer[1] },
  { source: inputLayer[2], target: hiddenLayer[2] },
  { source: hiddenLayer[0], target: outputLayer[0] },
  { source: hiddenLayer[0], target: outputLayer[1] },
  { source: hiddenLayer[1], target: outputLayer[0] },
  { source: hiddenLayer[1], target: outputLayer[1] },
  { source: hiddenLayer[2], target: outputLayer[0] },
  { source: hiddenLayer[2], target: outputLayer[1] },
];

svg
  .selectAll(".circle")
  .data([...inputLayer, ...hiddenLayer, ...outputLayer])
  .enter()
  .append("circle")
  .attr("class", "circle")
  .attr("cx", d => d.x)
  .attr("cy", d => d.y)
  .attr("r", 10);

svg
  .selectAll(".line")
  .data(links)
  .enter()
  .append("line")
  .attr("class", "line")
  .attr("x1", d => d.source.x)
  .attr("y1", d => d.source.y)
  .attr("x2", d => d.target.x)
  .attr("y2", d => d.target.y);
