import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';
import App from './App';

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

  render() {
    return (
      <div className="App">
        <header className="App-header">
          <App changeResults={this.changeResults} />
          {this.state.results != null &&
            <ul>
              {Object.keys(this.state.results).map((disease, i) => (
                <li key={i}>
                  <p> {disease}: {this.state.results[disease]}</p>
                </li>
              ))}
            </ul>
          }
        </header>
      </div>
    );
  }
}

export default StateApp;