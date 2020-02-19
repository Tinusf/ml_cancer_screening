import React, { Component } from 'react';
import './App.css';
import App from './App';
import Collapsible from 'react-collapsible';

class StateApp extends Component {
  constructor(props) {
    super(props);
    this.state = {
      results: null,
    };
  }

  changeResults = (results) => {
    this.setState({ results: results });
    console.log(results);
  }

  getMostLikely = () => {
    let highestVal = 0.0;
    let highestDisease = "";
    Object.keys(this.state.results).forEach((disease, i) => {
      let probability = this.state.results[disease];
      if (probability > highestVal) {
        highestVal = probability;
        highestDisease = disease;
      }
    });
    return highestDisease;
  }

  render() {
    return (
      <div className="App">
        <header className="App-header">
          <App changeResults={this.changeResults} />
          {this.state.results != null &&
            <div>
              <p>{this.getMostLikely()}</p>
              <Collapsible trigger="Advanced Results">
                <ul>
                  {Object.keys(this.state.results).map((disease, i) => (
                    <li key={i}>
                      <p> {disease}: {this.state.results[disease]}</p>
                    </li>
                  ))}
                </ul>
              </Collapsible>
            </div>
          }
        </header>
      </div>
    );
  }
}

export default StateApp;