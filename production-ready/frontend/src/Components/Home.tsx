import React from 'react';
import PdfUpload from './PdfUpload';
import Summary from './Summary';
import Header from './Header';
import Bot from './Main';
import { RagStateInterface } from '../ourtypes';
import './Home.css'

interface HomeProps {
  ragState: RagStateInterface;
  setRagState: React.Dispatch<React.SetStateAction<RagStateInterface>>;
}

const Home: React.FC<HomeProps> = ({ ragState, setRagState }) => {
  return (
    <div className="home-container">
      <Header />
      <div className="content">
        <div className="left-box">
          <PdfUpload ragState={ragState} setRagState={setRagState} />
          <Summary ragState={ragState} setRagState={setRagState} />
        </div>
        <div className="right-box">
          <Bot ragState={ragState} setRagState={setRagState} />
        </div>
      </div>
    </div>
  );
};

export default Home;
