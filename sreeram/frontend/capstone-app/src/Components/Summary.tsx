import React, { useState, useEffect } from 'react';
import AWS from 'aws-sdk';
import './Summary.css'
import { RagStateInterface } from '../ourtypes';

interface SummaryProps {
  ragState: RagStateInterface;
  setRagState: React.Dispatch<React.SetStateAction<RagStateInterface>>;
}

const Summary: React.FC<SummaryProps> = ({ ragState, setRagState }) => {
  const [summary, setSummary] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  useEffect(() => {
    console.log("summary Updated ragState:", ragState);
  }, [ragState]); // This will run every time ragState changes
  console.log("ragState",ragState)

  const ragsummarytxt = ragState.fileuuid.slice(0, -4);
  const summary_url = `https://capstoneragmodel.s3.amazonaws.com/summaries/${ragsummarytxt}.txt`

  const fetchSummary = async () => {
    try {
      console.log("summary_url",summary_url)
      const response = await fetch(summary_url);
      if (response.ok) {
        const text = await response.text();
        setSummary(text); // Set the summary after successful fetch
        setErrorMessage(null); // Clear any previous error message
      } else {
        console.error('Failed to fetch summary:', response.status, response.statusText);
        setErrorMessage('Failed to fetch the summary. Please try again.');
        setSummary(null);
      }
    } catch (error) {
      console.error('Error fetching the summary:', error);
      setErrorMessage('Error fetching the summary.');
      setSummary(null); // Clear the previous summary if there's an error
    }
  };

  return (
    <div className="summary-container">
      <h2>Request Summary</h2>
      <button onClick={fetchSummary}>Get Summary</button>
      {errorMessage && <p style={{ color: 'red' }}>{errorMessage}</p>}
      {summary && (
        <div>
          <h3>Summary:</h3>
          <p>{summary}</p>
        </div>
      )}
    </div>
  );
};

export default Summary;