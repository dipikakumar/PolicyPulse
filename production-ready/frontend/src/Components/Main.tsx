import React, { useState, useEffect } from 'react';
import { RagStateInterface } from '../ourtypes';
import ChatBot from "react-chatbotify";
import { log } from 'console';
import './Bot.css'

interface ChatBotProps {
  ragState: RagStateInterface;
  setRagState: React.Dispatch<React.SetStateAction<RagStateInterface>>;
}

const Bot: React.FC<ChatBotProps> = ({ ragState, setRagState }) => {
  const settings = {
    general: { embedded: true },
    chatHistory: { storageKey: "example_smart_conversation" },
    header: { 
      title: "Capstone Chatbot",
      showAvatar: true,
      avatar: "../assets/chatbot_image.png"
    }
  }

  async function fetchData(messageHistory: any) {
    try {
      console.log("got here");
      console.log("messageHistory", messageHistory);

      const payload = JSON.stringify({
        messages: messageHistory, // Send the message history to the backend
      });

      console.log("payload", payload);

      const response = await fetch('https://izkdlxtzx6.execute-api.us-east-1.amazonaws.com/dev/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: payload,
      });

      const data = await response.json();
      console.log("data", data);
      const parsedBody = JSON.parse(data.body);
      const reply = parsedBody.reply;
      console.log("reply",reply)

      // Update ragState with new messages from backend
      setRagState((prevState) => ({
        ...prevState,
        message_history: [...prevState.message_history, { role: 'bot', content: reply }],
      }));

      return reply; // Backend expected to return a 'reply' field
    } catch (error) {
      console.error(error);
      return "Oh no I don't know what to say!";
    }
  }

  // ChatBot flow configuration
  const flow = {
    start: {
      message: "Hey! Send me a message and I'll reply.",
      path: "loop",
    },
    loop: {
      message: async (params: any) => {
        console.log("params.userInput", params.userInput);

        // Add user input to the message history and send it to fetchData
        const userMessage = { role: 'user', content: params.userInput };
        setRagState((prevState) => ({
          ...prevState,
          message_history: [...prevState.message_history, userMessage],
        }));

        const result = await fetchData([...ragState.message_history, userMessage]);
        return result;
      },
      path: "loop",
    },
    end: {
      message: "Hey! Send me a message and I'll reply.",
      chatDisabled: true,
    },
  };

  return (
    <div>
      <ChatBot
        settings={settings}
        flow={flow}
        styles={{
          sendButtonStyle: { backgroundColor: "blue", color: "white" },
        }}
      />
    </div>
  );
};

export default Bot;
