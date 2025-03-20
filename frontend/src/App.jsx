import React from "react";
import Header from "./components/Header";
import LeftBox from "./components/LeftBox";
import RightBox from "./components/RightBox";

import BackgroundAnimation from "./components/BackgroundAnimation";

function App() {
  return (
    <div className="relative min-h-screen flex flex-col">
      <Header />
      <BackgroundAnimation />
      <div className="flex flex-grow items-center justify-between p-6 w-full relative z-10">
        <div className="w-1/2 flex justify-center">
          <LeftBox />
        </div>
        <div className="w-1/2 flex justify-center">
          <RightBox />
        </div>
      </div>
    </div>
  );
}

export default App;
