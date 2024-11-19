import React, { useState, useEffect } from 'react';
import AWS from 'aws-sdk';
import {RagStateInterface} from '../ourtypes'
import './PdfUpload.css'

const S3_BUCKET = 'capstoneragmodel';
const REGION = 'us-east-1';

AWS.config.update({
  accessKeyId: 'AKIAZ7SAKUX4I5RFWYGB',
  secretAccessKey: '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',
  region: REGION,
});

const s3 = new AWS.S3();

interface PdfUploadProps {
  ragState: RagStateInterface;
  setRagState: React.Dispatch<React.SetStateAction<RagStateInterface>>;
}

const PdfUpload: React.FC<PdfUploadProps> = ({ ragState, setRagState }) => {
  const [file, setFile] = useState<File | null>(null);
  const [message, setMessage] = useState<string>('');

  function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => 
      ((Math.random() * 16) | 0).toString(16).replace('x', c === 'x' ? '' : '8')
    );
  }
  

  useEffect(() => {
    console.log("Updated ragState:", ragState);
  }, [ragState]); // This will run every time ragState changes


  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const uuid = generateUUID();
    const fileuuid = `${uuid}.pdf`;
    const files = event.target.files; 
    console.log("files",files)
    console.log("if (files)", files ? true : false);

    if (files) {
      console.log("Here")
      const selectedFile = files[0];
      console.log("selectedFile",selectedFile)
      console.log("selectedFile.name",selectedFile.name)
      setFile(selectedFile); 
      setRagState(prevState => ({
        ...prevState,
        pdf_url: selectedFile.name, 
        fileuuid: fileuuid
      }));
    }
  };

  const uploadFile = async () => {
    if (!file) {
      setMessage('Please select a file to upload.');
      return;
    }

    const params = {
      Bucket: S3_BUCKET,
      Key: `pdf_uploads/${ragState.fileuuid}`,
      Body: file,
      ContentType: 'application/pdf',
    };

    try {
      await s3.upload(params).promise();
      setMessage('File uploaded successfully!');
    } catch (error) {
      setMessage(`Error uploading file`);
    }
  };
  return (
    <div className="upload-container">
        <h2 className="upload-title">Upload PDF to S3</h2>
        <input type="file" accept=".pdf" onChange={handleFileChange} className="upload-input" />
        <button onClick={uploadFile} className="upload-button">Upload</button>
        {message && <p className="upload-message">{message}</p>}
    </div>
);
};

export default PdfUpload;