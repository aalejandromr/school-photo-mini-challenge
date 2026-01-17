import { ShadowParams } from '../types';

interface LightControlsProps {
  params: ShadowParams;
  onParamsChange: (params: ShadowParams) => void;
}

export default function LightControls({ params, onParamsChange }: LightControlsProps) {
  const handleAngleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onParamsChange({
      ...params,
      lightAngle: parseFloat(e.target.value),
    });
  };

  const handleElevationChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onParamsChange({
      ...params,
      lightElevation: parseFloat(e.target.value),
    });
  };

  return (
    <div className="light-controls">
      <h3>Light Direction</h3>
      
      <div className="control-group">
        <label htmlFor="light-angle">
          Angle: {Math.round(params.lightAngle)}°
        </label>
        <input
          id="light-angle"
          type="range"
          min="0"
          max="360"
          step="1"
          value={params.lightAngle}
          onChange={handleAngleChange}
          className="slider"
        />
        <div className="control-value">{Math.round(params.lightAngle)}°</div>
      </div>

      <div className="control-group">
        <label htmlFor="light-elevation">
          Elevation: {Math.round(params.lightElevation)}°
        </label>
        <input
          id="light-elevation"
          type="range"
          min="0"
          max="90"
          step="1"
          value={params.lightElevation}
          onChange={handleElevationChange}
          className="slider"
        />
        <div className="control-value">{Math.round(params.lightElevation)}°</div>
      </div>
    </div>
  );
}
