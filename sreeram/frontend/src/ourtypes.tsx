import React from 'react';

// Interface for the state
export interface RagStateInterface {
    pdf_url: string;  // URL for the PDF
    fileuuid: string;
    message_history: { role: string, content: string }[];  // Array of message history
  }
  