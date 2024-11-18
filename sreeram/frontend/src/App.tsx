import React, { useState } from 'react';
import './App.css';
import Home from './Components/Home';
import { RagStateInterface } from './ourtypes';

function App() {
  // Define the initial ragState
  const initialState: RagStateInterface = {
    pdf_url: '',
    fileuuid: '',
    message_history: [
      { role: 'bot', content: 'You are a helpful chatbot.' },]
  };
  

  // Local state for ragState
  const [ragState, setRagState] = useState<RagStateInterface>(initialState);

  return (
    <div className="App">
      <Home ragState={ragState} setRagState={setRagState} />
    </div>
  );
}

export default App;
